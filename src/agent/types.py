"""Advisory pipeline output types for the ConflictAdvisoryAgent.

Research simulation only — not for operational use.

All types are frozen dataclasses (pure value objects). No mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.conflict.risk import compute_conflict_risk


@dataclass(frozen=True)
class PairwiseCPA:
    """Closest-point-of-approach geometry between two aircraft over a predicted horizon.

    Computed from predicted (not true) trajectories at a single prediction origin.

    Attributes
    ----------
    scenario_id : str
        Identifier of the parent scenario.
    aircraft_1 : str
        ACID of the first aircraft.
    aircraft_2 : str
        ACID of the second aircraft.
    origin_t : float
        Prediction origin time in seconds.
    cpa_lateral_nm : float
        Minimum predicted lateral separation (NM) over the full horizon.
    cpa_vertical_ft : float
        Predicted vertical separation (ft) at the time of minimum lateral distance.
    cpa_time_s : float
        Absolute time (seconds) of closest predicted approach.
    time_to_cpa_s : float
        Seconds from ``origin_t`` to ``cpa_time_s``.
    closure_rate_nm_per_s : float
        Rate of change of lateral distance at the prediction origin (NM/s),
        positive = closing, negative = diverging.
    """

    scenario_id: str
    aircraft_1: str
    aircraft_2: str
    origin_t: float
    cpa_lateral_nm: float
    cpa_vertical_ft: float
    cpa_time_s: float
    time_to_cpa_s: float
    closure_rate_nm_per_s: float


@dataclass(frozen=True)
class ConflictAdvisory:
    """Advisory signal for a single (aircraft-pair, prediction-origin) sample.

    Attributes
    ----------
    cpa : PairwiseCPA
        CPA geometry backing this advisory.
    is_conflict : bool
        True when the predicted CPA violates both the lateral and vertical
        separation minima simultaneously.
    p_risk : float
        Continuous risk score in [0, 1]. 0 = no risk; 1 = maximum risk.
        Defined as ``min(lat_risk, vert_risk)`` where each axis risk decays
        linearly from 1 at zero separation to 0 at the relevant minimum.
    lead_time_s : float | None
        Seconds from ``origin_t`` to the first predicted LoS instant.
        ``None`` when ``is_conflict`` is False.
    """

    cpa: PairwiseCPA
    is_conflict: bool
    p_risk: float
    lead_time_s: float | None


@dataclass(frozen=True)
class ScenarioAdvisoryReport:
    """Aggregated advisory output for a full scenario across all prediction origins.

    Attributes
    ----------
    scenario_id : str
        Identifier of the evaluated scenario.
    num_aircraft : int
        Number of distinct aircraft in the scenario.
    num_origins : int
        Number of prediction origins evaluated.
    advisories : tuple[ConflictAdvisory, ...]
        All (pair, origin) advisory records (including non-conflict ones).
    flagged_pairs : frozenset[tuple[str, str]]
        Unordered pairs ``(acid_1, acid_2)`` where at least one origin was
        flagged as a conflict.
    max_p_risk : float
        Highest ``p_risk`` across all advisories in this scenario.
    earliest_lead_time_s : float | None
        Smallest lead time across all flagged advisories; ``None`` if none flagged.
    """

    scenario_id: str
    num_aircraft: int
    num_origins: int
    advisories: tuple[ConflictAdvisory, ...]
    flagged_pairs: frozenset[tuple[str, str]]
    max_p_risk: float
    earliest_lead_time_s: float | None

    @property
    def has_conflict(self) -> bool:
        """Return True if any advisory is flagged as a conflict."""
        return len(self.flagged_pairs) > 0

    @property
    def num_flagged(self) -> int:
        """Return count of advisories with is_conflict=True."""
        return sum(1 for a in self.advisories if a.is_conflict)
