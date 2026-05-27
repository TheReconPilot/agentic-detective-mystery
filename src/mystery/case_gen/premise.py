"""Seed-driven premise rolling.

Small instruct models have a heavy prior on "Edwardian manor + butler/maid/cook"
when asked for a murder mystery, even with `temperature=0.7` and varying seeds.
The seed nudges the sampler but cannot escape the genre attractor named in the
prompt. Empirically, three independently-seeded cases all ended up in a study.

We fix this by *deciding* the high-level scenario in pure Python before the LLM
ever sees a prompt: a deterministic `random.Random(seed)` picks a setting, era,
cast template, and death scenario from curated lists. The LLM is then handed
those choices as hard constraints rather than left to fall back on its prior.

This keeps generation reproducible from a seed (`Random(seed)` is stable) while
forcing the case space wide open.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """A bundle of compatible setting + era + plausible roles + death types.

    Roles and death scenarios are grouped per setting so the roll never
    produces nonsense like "a sommelier on a Mars colony". Each scenario
    lists at least 5 candidate roles so the sampler can vary the cast.
    """

    setting: str
    era: str
    cast_roles: list[str]
    death_scenarios: list[str]


# Twenty deliberately heterogeneous scenarios. Add or edit freely — the only
# constraint is internal consistency within each entry (roles plausible for
# setting; death scenario plausible for setting).
SCENARIOS: list[Scenario] = [
    Scenario(
        setting="a Series-B biotech startup's downtown headquarters, late on a Friday",
        era="present day",
        cast_roles=[
            "the founder-CEO",
            "the chief scientist",
            "the lead venture investor",
            "the head of HR",
            "a recently-terminated lab tech",
            "the night-shift security guard",
        ],
        death_scenarios=[
            "found slumped at a workstation, apparently poisoned by something in their coffee",
            "found electrocuted in the cold-storage server room",
            "found strangled in the corner office, the door locked from inside",
        ],
    ),
    Scenario(
        setting="a remote alpine ski lodge cut off by a three-day blizzard",
        era="present day",
        cast_roles=[
            "the lodge owner",
            "a celebrity ski instructor",
            "a wealthy hedge-fund guest",
            "the new sous-chef",
            "an off-duty homicide detective on vacation",
            "the lodge's longtime handyman",
        ],
        death_scenarios=[
            "found at the bottom of the great staircase, neck broken",
            "found in the sauna, the door wedged shut from outside",
            "found in the boot room with a ski pole through their chest",
        ],
    ),
    Scenario(
        setting="an isolated Antarctic research station mid-winter",
        era="present day",
        cast_roles=[
            "the station chief",
            "the lead glaciologist",
            "the comms officer",
            "the cook-medic",
            "a visiting climate-policy journalist",
            "the maintenance engineer",
        ],
        death_scenarios=[
            "found face-down on the ice outside the airlock with no coat on",
            "found in the generator room with a wrench through their skull",
            "found unresponsive in their bunk, an empty bottle of sleeping pills beside them",
        ],
    ),
    Scenario(
        setting="a championship-bound esports team's shared training house",
        era="present day",
        cast_roles=[
            "the team's star carry player",
            "the head coach",
            "the team manager",
            "the substitute rifler who just got benched",
            "a streamer girlfriend visiting for the weekend",
            "the team's nutritionist",
        ],
        death_scenarios=[
            "found dead at their gaming rig, an energy drink can on the desk",
            "found drowned in the basement plunge pool",
            "found dead on the practice floor with a snapped keyboard cable around their neck",
        ],
    ),
    Scenario(
        setting="a Michelin-starred restaurant kitchen ten minutes after the last cover",
        era="present day",
        cast_roles=[
            "the head chef",
            "the sous-chef",
            "the maître d'",
            "the sommelier",
            "the head of the produce supplier the chef just publicly humiliated",
            "the pastry chef",
        ],
        death_scenarios=[
            "found dead at the pass, face in a sauce, foaming at the mouth",
            "found behind the walk-in with a fish-knife between their ribs",
            "found in the dry-store room, head bashed in by a marble pestle",
        ],
    ),
    Scenario(
        setting="a Buenos Aires tango academy during a national championship weekend",
        era="1953",
        cast_roles=[
            "the academy's grande dame",
            "the visiting champion from Paris",
            "the bandoneón player whose tune everyone steps to",
            "the rival academy's spy",
            "a faded former champion turned bartender",
            "the academy's bookkeeper",
        ],
        death_scenarios=[
            "found dead in the dressing room mirror chair, garrotted with a silk scarf",
            "found at the foot of the rehearsal-room ladder, neck broken",
            "found in the alley behind the academy with three knife wounds",
        ],
    ),
    Scenario(
        setting="a medieval Iberian monastery library on the night of the abbot's election",
        era="the year 1387",
        cast_roles=[
            "the librarian-monk",
            "the abbot-elect",
            "the abbey's physician",
            "the visiting Cistercian inquisitor",
            "a novice who arrived only last spring",
            "the abbey's cellarer",
        ],
        death_scenarios=[
            "found dead among the scriptorium desks, an inkwell of strange black liquid spilled",
            "found at the bottom of the bell-tower stairs",
            "found in the herb-garden cell, lips stained dark from something he chewed",
        ],
    ),
    Scenario(
        setting="a Mars colony habitat dome on its fifty-third sol",
        era="the late twenty-first century",
        cast_roles=[
            "the colony commander",
            "the hydroponics lead",
            "the chief medical officer",
            "the comms officer with a frayed connection to Earth",
            "the geologist who keeps disappearing into the dust",
            "the rover mechanic",
        ],
        death_scenarios=[
            "found dead in the airlock with their suit's oxygen regulator tampered with",
            "found in the hydroponics bay, drowned in a nutrient tank",
            "found in their quarters, blunt-force trauma from a sample drill",
        ],
    ),
    Scenario(
        setting="a touring off-Broadway theatre company during intermission of opening night",
        era="1974",
        cast_roles=[
            "the lead actor on the verge of being recast",
            "the director who built the company from nothing",
            "the understudy nobody expected to need",
            "the stage manager",
            "the playwright who flew in for the opening",
            "the lighting tech with a grudge",
        ],
        death_scenarios=[
            "found dead in the green room, a glass of something amber still in their hand",
            "found in the rigging loft, a sandbag rope cut through",
            "found in their dressing room, makeup mirror smashed, throat cut",
        ],
    ),
    Scenario(
        setting="an oceanographic research vessel two thousand miles from the nearest port",
        era="present day",
        cast_roles=[
            "the chief scientist",
            "the ship's captain",
            "the deep-submersible pilot",
            "the marine biologist who just made a career-defining discovery",
            "the documentary filmmaker embedded for the season",
            "the bosun",
        ],
        death_scenarios=[
            "found in the wet lab, head split open by what looks like a winch hook",
            "found face-down in the dive pool with the gate locked",
            "found in their cabin, an injection mark on the inner elbow",
        ],
    ),
    Scenario(
        setting="a Cold War-era diesel submarine on a sixty-day patrol under the Arctic ice",
        era="1983",
        cast_roles=[
            "the captain",
            "the executive officer",
            "the political officer",
            "the chief engineer",
            "the cook with secrets",
            "the youngest sonar operator on his first patrol",
        ],
        death_scenarios=[
            "found in the torpedo room with a knife in his back",
            "found dead in the head, a wrench beside him",
            "found suffocated in his bunk, his face cyanotic",
        ],
    ),
    Scenario(
        setting="a Victorian-era travelling circus's winter quarters",
        era="1893",
        cast_roles=[
            "the ringmaster",
            "the high-wire performer who took the headlining act last season",
            "the strongman",
            "the bearded fortune-teller everyone half-believes",
            "the new acrobat hired in Marseilles",
            "the lion handler",
        ],
        death_scenarios=[
            "found in the lion's cage at dawn, but not — the coroner is sure — killed by the lion",
            "found in the costume wagon, hanged from a trapeze rope",
            "found in the empty big top with a sword-swallower's blade through them",
        ],
    ),
    Scenario(
        setting="a Tokyo capsule hotel's overnight shift",
        era="present day",
        cast_roles=[
            "the night manager",
            "a salaryman who hasn't been home in eight days",
            "an off-duty police officer who checked in alone",
            "an American backpacker in Tokyo for the first time",
            "the cleaner",
            "a tattoo-shop apprentice in the city for a convention",
        ],
        death_scenarios=[
            "found dead inside their pod, the privacy curtain drawn",
            "found in the public bath, drowned in two feet of water",
            "found in the alley by the staff entrance with their wallet untouched",
        ],
    ),
    Scenario(
        setting="a remote African archaeological dig the night a major find was announced",
        era="1932",
        cast_roles=[
            "the lead British archaeologist",
            "the rival French archaeologist who arrived 'just to observe'",
            "the local foreman who actually found the chamber",
            "the wealthy benefactor visiting from London",
            "the medical doctor accompanying the expedition",
            "the photographer documenting the dig",
        ],
        death_scenarios=[
            "found dead inside the newly-opened tomb, a spear-tipped fragment in his neck",
            "found at the foot of the excavation ladder, neck broken",
            "found dead in his tent, the mosquito net cut and a vial broken on the floor",
        ],
    ),
    Scenario(
        setting="a weekend tech-accelerator retreat at a mountain lodge",
        era="present day",
        cast_roles=[
            "the accelerator's managing partner",
            "the founder of the cohort's star company",
            "the founder of the cohort's most-pivoted-on company",
            "a guest speaker — a famously cantankerous unicorn CEO",
            "the lodge's own caretaker",
            "the program manager",
        ],
        death_scenarios=[
            "found dead in the hot tub on the deck, lips faintly blue",
            "found at the bottom of the property's pier, weights tied to their ankles",
            "found in the boardroom, throat cut, the whiteboard wiped clean",
        ],
    ),
    Scenario(
        setting="a small-town AM radio station during its overnight shift",
        era="1991",
        cast_roles=[
            "the overnight DJ",
            "the station owner who came in unannounced",
            "the engineer fixing the failing transmitter",
            "a regular caller who finally drove down in person",
            "the morning-show host who arrived early",
            "the news intern pulling a double",
        ],
        death_scenarios=[
            "found dead in the booth, headphones still on, the carrier wave still humming",
            "found in the parking lot beside their idling car",
            "found in the record library, head bashed in by a vinyl crate",
        ],
    ),
    Scenario(
        setting="a private boarding school dormitory during final-exam week",
        era="present day",
        cast_roles=[
            "the dorm proctor",
            "the school's top student",
            "the legacy heir who has been failing quietly",
            "the new transfer student",
            "the AP-history teacher who lives on-site",
            "the night-shift custodian",
        ],
        death_scenarios=[
            "found dead in the dorm shower, the water still running",
            "found at the foot of the dorm stairwell with their notes scattered",
            "found in the empty library, an unmarked pill bottle on the table",
        ],
    ),
    Scenario(
        setting="a remote lighthouse on a small island during a multi-day storm",
        era="1924",
        cast_roles=[
            "the head keeper",
            "the assistant keeper, transferred in last month",
            "the keeper's wife",
            "a shipwrecked sailor sheltering since dawn",
            "the supply boatman, stranded by the storm",
            "the visiting inspector from the Lighthouse Board",
        ],
        death_scenarios=[
            "found at the foot of the spiral staircase, neck broken",
            "found drowned in the cistern at the base of the tower",
            "found dead on the gallery deck, half-frozen, the lamp gone out",
        ],
    ),
    Scenario(
        setting="a haute-couture fashion house twenty-four hours before its Paris show",
        era="present day",
        cast_roles=[
            "the founding designer",
            "the heir-apparent creative director",
            "the head seamstress of thirty years",
            "the celebrity muse who flew in this morning",
            "the brand's CFO",
            "the assistant who knows where every body is buried",
        ],
        death_scenarios=[
            "found dead at the cutting table, pinking shears in their chest",
            "found dead in the fitting room, hung with a length of silk",
            "found in the archive, asphyxiated by a plastic garment-cover",
        ],
    ),
    Scenario(
        setting="a Las Vegas casino's high-roller poker room at four in the morning",
        era="present day",
        cast_roles=[
            "the pit boss",
            "the casino's biggest whale of the month",
            "a professional player on a six-day heater",
            "the floor's senior dealer",
            "a regulator quietly checking the room",
            "the cocktail server who saw everything",
        ],
        death_scenarios=[
            "found dead in the cashier cage, the room's chips scattered around them",
            "found dead in the private bathroom off the high-roller suite",
            "found garrotted in the parking garage beside their car",
        ],
    ),
]


@dataclass(frozen=True)
class Premise:
    """Concrete, seed-derived scaffolding handed to the case-generation LLM.

    `cast_roles` is a list of 3-5 short role labels; the LLM must produce one
    suspect per entry, mapping role → archetype → name → motive. Pre-deciding
    these in Python is what stops the model from defaulting to butler/maid/cook.
    """

    setting: str
    era: str
    cast_roles: list[str]
    death_scenario: str

    def to_constraint_text(self) -> str:
        """Render as a hard-constraint block for the user prompt."""
        roles_block = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(self.cast_roles))
        return (
            f"Setting: {self.setting}\n"
            f"Era: {self.era}\n"
            f"Cast — each suspect MUST map to exactly one of these roles "
            f"(use as the suspect's `archetype`, in any order):\n"
            f"{roles_block}\n"
            f"Death: the victim is {self.death_scenario}.\n"
            f"Build locations, clues, and timeline that fit this setting — "
            f"do not default to manor-house tropes (no butlers, maids, "
            f"libraries, or studies unless the setting above genuinely calls for them)."
        )


def roll_premise(seed: int) -> Premise:
    """Pick a scenario, sample a cast of 3-5 roles, choose a death scenario.

    Deterministic in ``seed``: `random.Random(seed)` is stable across Python
    versions for the operations used here (choice, randint, sample), so calling
    twice with the same seed yields the same Premise. That property matters
    because the bible's `seed` field is also the case id on disk — re-running
    `mystery new --seed N` should land on the same case.
    """
    rng = random.Random(seed)
    scenario = rng.choice(SCENARIOS)
    cast_size = rng.randint(3, min(5, len(scenario.cast_roles)))
    cast_roles = rng.sample(scenario.cast_roles, cast_size)
    death = rng.choice(scenario.death_scenarios)
    return Premise(
        setting=scenario.setting,
        era=scenario.era,
        cast_roles=cast_roles,
        death_scenario=death,
    )
