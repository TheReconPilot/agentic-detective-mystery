"""Tests for the seed-driven premise roll.

The diversity claim of the new generation pipeline rests on these properties:

  - ``roll_premise`` is deterministic in the seed (same seed → same premise),
  - close seeds (1, 2, 3, …) reach a wide range of scenarios (not all "manor"),
  - every Scenario in the catalogue is internally sane (enough roles, at least
    one death scenario).

If any of these break, the diagnosis in the user's complaint will reappear.
"""

from __future__ import annotations

from mystery.case_gen.premise import SCENARIOS, Premise, roll_premise


def test_roll_premise_is_deterministic() -> None:
    """Same seed must always yield the same Premise."""
    a = roll_premise(42)
    b = roll_premise(42)
    assert a == b


def test_roll_premise_varies_across_close_seeds() -> None:
    """Twenty contiguous seeds should hit several distinct scenarios.

    The whole point of this module is that 0..19 are not all the same
    "study murder in a manor". We require at least 8 unique settings — well
    under the 20-scenario catalogue ceiling, but a real sanity check that
    the rolls are spreading out across the table.
    """
    settings = {roll_premise(s).setting for s in range(20)}
    assert len(settings) >= 8, f"only {len(settings)} unique settings in seeds 0..19: {settings}"


def test_roll_premise_returns_3_to_5_cast_roles() -> None:
    """Cast size matches the suspects[].count constraint in SYSTEM_PROMPT."""
    for seed in range(30):
        premise = roll_premise(seed)
        assert 3 <= len(premise.cast_roles) <= 5, (
            f"seed={seed} produced {len(premise.cast_roles)} roles"
        )


def test_roll_premise_cast_roles_are_unique() -> None:
    """Sampling without replacement: every rolled role appears at most once."""
    for seed in range(30):
        premise = roll_premise(seed)
        assert len(premise.cast_roles) == len(set(premise.cast_roles)), (
            f"seed={seed} produced duplicate cast roles: {premise.cast_roles}"
        )


def test_roll_premise_cast_roles_come_from_their_scenario() -> None:
    """Roles must be drawn from the scenario the premise reports, not crossed.

    Mixing roles across scenarios would produce nonsense like a sommelier on
    a Mars colony — exactly the kind of incoherence the per-scenario role
    lists are meant to prevent.
    """
    setting_to_roles = {sc.setting: set(sc.cast_roles) for sc in SCENARIOS}
    for seed in range(30):
        premise = roll_premise(seed)
        valid_roles = setting_to_roles[premise.setting]
        for role in premise.cast_roles:
            assert role in valid_roles, (
                f"seed={seed} role {role!r} not in scenario {premise.setting!r}"
            )


def test_to_constraint_text_mentions_every_role() -> None:
    """The text emitted into the user prompt must list every rolled role.

    If a role were dropped, the model would silently fall back to its prior
    for that slot — defeating the whole exercise.
    """
    premise = roll_premise(7)
    text = premise.to_constraint_text()
    for role in premise.cast_roles:
        assert role in text, f"role {role!r} missing from constraint text"
    assert premise.setting in text
    assert premise.era in text
    assert premise.death_scenario in text


def test_to_constraint_text_warns_against_manor_defaults() -> None:
    """The constraint block must actively repel butler/maid/library defaults.

    This is the one piece of negative guidance that survives even though the
    user opted out of broad negative-prompt scaffolding — it is specifically
    tied to the diversity problem, not to general genre restriction.
    """
    text = roll_premise(0).to_constraint_text()
    assert "butler" in text.lower() or "manor" in text.lower(), (
        "constraint text should explicitly warn against the manor attractor"
    )


def test_every_scenario_has_enough_roles_and_deaths() -> None:
    """SCENARIOS catalogue invariant: each entry must support a full roll.

    With min cast of 3 and a death pick, a scenario with fewer than 3 roles
    or zero death scenarios would crash ``roll_premise`` for some seed.
    """
    for sc in SCENARIOS:
        assert len(sc.cast_roles) >= 5, (
            f"scenario {sc.setting!r} has only {len(sc.cast_roles)} roles; need >=5"
        )
        assert len(sc.death_scenarios) >= 1, f"scenario {sc.setting!r} has no death scenarios"


def test_premise_dataclass_is_frozen() -> None:
    """Premise is treated as a value; the rolled scenario shouldn't mutate."""
    premise = Premise(setting="x", era="y", cast_roles=["a", "b", "c"], death_scenario="z")
    try:
        premise.setting = "different"  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("Premise should be frozen")
