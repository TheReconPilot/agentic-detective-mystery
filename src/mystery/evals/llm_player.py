"""LLM-driven detective playtester.

Unlike :mod:`mystery.evals.optimal_player`, which DFSes the bible to win by
construction, this player only sees what the game would show a human: the
current room, the visible exits, the cast of suspects, the notebook, and the
last tool output. It then has to *decide* what to do next via an LLM call.

That makes it the only test surface that exercises the full chain:

* the router's tolerance to natural-language commands the LLM emits,
* the observation we render each turn (does it contain enough to play?),
* the suspect agent's grounding under non-canned questions,
* whether a generated case is solvable without privileged knowledge.

Bugs that show up here cannot be caught by canned ``FakeListChatModel`` tests.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage, SystemMessage

from mystery.graph.game import build_game_graph
from mystery.graph.router import ParseError, parse_action
from mystery.graph.state import GameState, initial_state

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.graph.state import Action
    from mystery.models import CaseBible


_DETECTIVE_SYSTEM = """\
You are the player in a text-based detective game. Your goal is to identify
the killer and accuse them. You must output exactly ONE command per turn,
no commentary, no markdown, no quotes — just the raw command line.

Available commands (you MUST use these exact verbs):
  move <location_id>            move to a connected location
  ask <suspect_id> <question>   interrogate a suspect (question can be many words)
  show <suspect_id> <clue_id>   confront a suspect with a specific clue you have found
  examine                       look around the current location for clues
  notes                         show your notebook (free action)
  accuse <suspect_id>           end the game by naming the killer

Strategy (follow in order):
1. If the current room is NOT YET examined, your next command is `examine`.
2. If there are still rooms in ROOMS NOT YET SEARCHED, your next command is
   `move <one of them>` — use the NEXT HOP TO UNSEARCHED ROOMS block to
   pick the right first step (the LLM cannot teleport).
3. Once every room is searched and CLUES FOUND is non-empty, the
   INTERROGATION phase begins:
   - First, `ask` each suspect ONCE about their whereabouts at the time of
     death. That establishes a baseline alibi.
   - Then, for each clue in CLUES FOUND, pick the suspect whose alibi or
     archetype it most plausibly contradicts and `show <suspect_id> <clue_id>`.
     This is where lies break — confrontation is the only way to make
     progress in this phase.
   - Do NOT ask the same suspect more than 2-3 questions back-to-back; that
     loops without producing new evidence. Rotate suspects or pivot to `show`.
4. When the evidence points clearly to one suspect, `accuse <suspect_id>`.
   The game ends instantly on accusation, right or wrong — don't guess, but
   DO accuse before turns run out; an indecisive detective loses.

Hard rules:
- NEVER `examine` once ROOMS NOT YET SEARCHED is empty. The next command
  MUST be `ask`, `show`, `accuse`, or (rarely) `move`. Re-examining a
  searched room is the single biggest failure mode; do not do it.
- NEVER repeat the same command twice in a row.
- After 3 consecutive `ask` commands without progress, your NEXT command
  MUST be `show <suspect> <clue>` or `accuse <suspect>`. Asking variations
  of the same question is not progress.
- Use ONLY the ids shown in the observation (snake_case). Do not invent ids.
- Write `ask` questions as natural English sentences, not snake_case_strings.
- For `show`, the clue id is one of the snake_case ids in CLUES FOUND.

Output format: ONE command line. Nothing else.
"""


@dataclass
class PlaytestStep:
    turn: int
    observation: str
    raw_command: str
    parsed_kind: str  # action kind, or "parse_error"
    output: str


@dataclass
class PlaytestReport:
    seed: int
    success: bool
    turns_used: int
    accused: str
    actual_killer: str
    parse_errors: int
    repeated_actions: int
    steps: list[PlaytestStep] = field(default_factory=list)
    suspect_commitments: dict[str, list[str]] = field(default_factory=dict)

    def as_text(self) -> str:
        head = (
            f"seed={self.seed} success={self.success} turns={self.turns_used} "
            f"accused={self.accused or '(none)'} actual={self.actual_killer} "
            f"parse_errors={self.parse_errors} repeats={self.repeated_actions}"
        )
        body = "\n".join(
            f"  t{s.turn} [{s.parsed_kind}] {s.raw_command!r}\n"
            f"      -> {s.output.splitlines()[0][:120] if s.output else ''}"
            for s in self.steps
        )
        return f"{head}\n{body}"


def _next_hop_toward(
    bible: CaseBible,
    start: str,
    targets: list[str],
) -> dict[str, str]:
    """BFS from ``start``; return ``{target: first_step}`` for each reachable target.

    The 14b LLM sees the FULL MAP but routinely tries to issue a single-hop
    move to a non-adjacent room. A precomputed "to reach X, first move to Y"
    hint sidesteps that whole class of failure — the model only has to read
    a label, not solve graph traversal.
    """
    if not targets:
        return {}
    adj: dict[str, list[str]] = {
        loc.id: list(loc.connected_location_ids) for loc in bible.locations
    }
    target_set = set(targets)
    parent: dict[str, str] = {start: start}
    q: deque[str] = deque([start])
    while q:
        node = q.popleft()
        for nxt in adj.get(node, []):
            if nxt in parent:
                continue
            parent[nxt] = node
            q.append(nxt)
    out: dict[str, str] = {}
    for target in target_set:
        if target not in parent or target == start:
            continue
        cur = target
        while parent[cur] != start:
            cur = parent[cur]
        out[target] = cur
    return out


def _recent_ask_streak(recent_kinds: list[str]) -> int:
    """Count trailing consecutive `interrogate` kinds in the action history."""
    streak = 0
    for kind in reversed(recent_kinds):
        if kind == "interrogate":
            streak += 1
        else:
            break
    return streak


def render_observation(
    state: GameState,
    bible: CaseBible,
    *,
    max_turns: int | None = None,
    recent_action_kinds: list[str] | None = None,
) -> str:
    """Render the textual state the LLM detective sees each turn.

    Deliberately bible-redacted: it shows what a player at this state would
    legitimately know (current room, exits, suspect roster, revealed clues,
    last tool output). It does NOT leak the killer, deception policies,
    motives, alibi truth values, or the canonical timeline.

    Surfaces *which* rooms have not yet been visited so a confused detective
    is nudged to explore — early playtests showed the LLM happily looping on
    `examine` in a fully-searched room without trying other locations.
    """
    here = next(loc for loc in bible.locations if loc.id == state["current_location_id"])
    exits = ", ".join(here.connected_location_ids) or "(none)"
    # Full map: the player legitimately knows the manor's layout. Without it the
    # LLM detective gets stranded trying to reach non-adjacent rooms in one move.
    map_lines = "\n".join(
        f"    {loc.id} <-> {', '.join(loc.connected_location_ids) or '(dead end)'}"
        for loc in bible.locations
    )
    suspect_ids = ", ".join(s.id for s in bible.suspects)
    revealed = ", ".join(state["revealed_clue_ids"]) or "(none yet)"
    notes_block = (
        "\n".join(f"  - {line}" for line in state["notebook"]) if state["notebook"] else "  (empty)"
    )
    last = state["last_output"] or "(none)"
    all_locs = {loc.id for loc in bible.locations}
    unexamined = sorted(all_locs - set(state["examined_location_ids"]))
    unexamined_str = ", ".join(unexamined) or "(none — you have searched everywhere)"
    here_examined = "yes" if here.id in state["examined_location_ids"] else "NO, not yet"
    next_hops = _next_hop_toward(bible, here.id, unexamined)
    if next_hops:
        hop_lines = "\n".join(
            f"    to reach {target}, first `move {step}`"
            for target, step in sorted(next_hops.items())
        )
        next_hop_block = f"NEXT HOP TO UNSEARCHED ROOMS (from here):\n{hop_lines}\n"
    else:
        next_hop_block = ""

    # Pattern-break nudge. The 14b detective will happily ask the same
    # suspect 12 questions in a row even with a "stop asking" rule in the
    # system prompt — autoregressive momentum drowns the constraint. The
    # observation-level call-out runs much closer to the model's next-token
    # decision and seems to be what actually shifts the action choice.
    ask_streak = _recent_ask_streak(recent_action_kinds or [])
    if ask_streak >= 3 and state["revealed_clue_ids"]:
        nudge_block = (
            f"\n!!! STOP: you have issued {ask_streak} consecutive `ask` commands "
            f"without making progress. Do NOT issue another `ask`. Your next "
            f"command MUST be either `show <suspect> <clue>` (confront someone "
            f"with one of your {len(state['revealed_clue_ids'])} clues) or "
            f"`accuse <suspect>` (commit to the killer).\n"
        )
    else:
        nudge_block = ""

    return (
        f"VICTIM: {bible.victim.name} ({bible.victim.role}), "
        f"found in {bible.victim.location_of_death_id} at t={bible.victim.time_of_death}.\n"
        f"YOU ARE IN: {here.id} — {here.name}. {here.description}\n"
        f"  examined this room? {here_examined}\n"
        f"EXITS (from here, single move): {exits}\n"
        f"FULL MAP (you must chain moves through adjacent rooms):\n{map_lines}\n"
        f"{next_hop_block}"
        f"SUSPECTS (use these ids): {suspect_ids}\n"
        f"CLUES FOUND: {revealed}\n"
        f"ROOMS NOT YET SEARCHED (need `examine` after moving): {unexamined_str}\n"
        f"NOTEBOOK:\n{notes_block}\n"
        f"LAST RESULT: {last}\n"
        f"TURN: {state['turn_count']}"
        f"{f' of {max_turns} (commit soon!)' if max_turns else ''}\n"
        f"{nudge_block}"
        f"\nWhat is your next command?"
    )


def _ask_detective(chat_model: BaseChatModel, observation: str) -> str:
    messages = [
        SystemMessage(content=_DETECTIVE_SYSTEM),
        HumanMessage(content=observation),
    ]
    response = chat_model.invoke(messages)
    raw = str(response.content).strip()
    # Take only the first non-empty line, in case the model can't help itself.
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "//", "<")):
            # Strip surrounding quotes / backticks if any.
            return line.strip("`'\" ")
    return raw


def _progress_score(state: GameState) -> int:
    """A monotonic 'detective is making progress' counter.

    Increases when the player examines a new room or reveals a new clue.
    Used as the no-progress abort signal: a small LLM detective will happily
    shuttle between two rooms without ever examining, and consecutive-repeat
    detection misses alternating cycles.
    """
    return len(state["examined_location_ids"]) + len(state["revealed_clue_ids"])


def play_with_llm(
    bible: CaseBible,
    vectorstore: Chroma,
    suspect_chat_model: BaseChatModel,
    detective_chat_model: BaseChatModel,
    *,
    max_turns: int = 60,
    max_consecutive_repeats: int = 4,
    no_progress_window: int = 12,
    commitment_extractor: CommitmentExtractor | None = None,
) -> PlaytestReport:
    """Run a full LLM-vs-LLM playtest. Returns a structured report.

    ``max_consecutive_repeats`` is a guardrail: the LLM detective will
    occasionally lock into a loop (e.g. re-examining a searched room) and
    consume every remaining turn. Once that many *consecutive identical*
    commands have been issued, abort and mark the case unsolved — that's a
    real failure to report, not an excuse to keep spending tokens.

    ``no_progress_window`` is a second guardrail for alternating-cycle loops
    (e.g. move A → move B → move A → …) that consecutive-repeat detection
    misses. If the progress score does not increase for that many turns,
    abort. Set high enough that legitimate "all rooms searched, now I'm
    interrogating" phases can run.
    """
    graph = build_game_graph(bible, vectorstore, suspect_chat_model, commitment_extractor)
    state = initial_state(bible)
    steps: list[PlaytestStep] = []
    parse_errors = 0
    repeated_actions = 0
    last_raw: str | None = None
    consecutive_repeats = 0
    consecutive_parse_errors = 0
    best_progress = _progress_score(state)
    turns_since_progress = 0
    recent_action_kinds: list[str] = []

    while not state["done"] and state["turn_count"] < max_turns:
        observation = render_observation(
            state,
            bible,
            max_turns=max_turns,
            recent_action_kinds=recent_action_kinds[-6:],
        )
        raw = _ask_detective(detective_chat_model, observation)

        if raw == last_raw:
            repeated_actions += 1
            consecutive_repeats += 1
        else:
            consecutive_repeats = 0
        last_raw = raw
        if consecutive_repeats >= max_consecutive_repeats:
            # Aborted loop: record it so the report is honest about why we stopped.
            steps.append(
                PlaytestStep(
                    turn=state["turn_count"],
                    observation=observation,
                    raw_command=raw,
                    parsed_kind="aborted_loop",
                    output=f"detective issued {raw!r} {consecutive_repeats + 1}x — abort",
                ),
            )
            break

        parsed: Action | ParseError = parse_action(raw)
        if isinstance(parsed, ParseError):
            parse_errors += 1
            consecutive_parse_errors += 1
            steps.append(
                PlaytestStep(
                    turn=state["turn_count"],
                    observation=observation,
                    raw_command=raw,
                    parsed_kind="parse_error",
                    output=parsed.message,
                ),
            )
            state = cast(
                "GameState",
                {**state, "last_output": f"PARSE ERROR: {parsed.message}"},
            )
            if consecutive_parse_errors >= 5:
                # Detective is hopelessly confused; abort to keep eval bounded.
                break
            continue

        consecutive_parse_errors = 0
        state["pending_action"] = parsed
        state = cast("GameState", graph.invoke(state))
        recent_action_kinds.append(parsed.kind)
        steps.append(
            PlaytestStep(
                turn=state["turn_count"],
                observation=observation,
                raw_command=raw,
                parsed_kind=parsed.kind,
                output=state["last_output"],
            ),
        )

        progress = _progress_score(state)
        if progress > best_progress:
            best_progress = progress
            turns_since_progress = 0
        else:
            turns_since_progress += 1
        if turns_since_progress >= no_progress_window and not state["done"]:
            steps.append(
                PlaytestStep(
                    turn=state["turn_count"],
                    observation="",
                    raw_command="",
                    parsed_kind="aborted_no_progress",
                    output=(
                        f"no new clues or examined rooms for {turns_since_progress} turns "
                        f"— aborting"
                    ),
                ),
            )
            break

    accusation = state["accusation"]
    commitments_view = {
        suspect_id: [c.summary for c in cs]
        for suspect_id, cs in state["suspect_commitments"].items()
    }
    return PlaytestReport(
        seed=bible.seed,
        success=bool(accusation and accusation.correct),
        turns_used=state["turn_count"],
        accused=accusation.accused_id if accusation else "",
        actual_killer=bible.killer_id,
        parse_errors=parse_errors,
        repeated_actions=repeated_actions,
        steps=steps,
        suspect_commitments=commitments_view,
    )
