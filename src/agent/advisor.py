"""ConflictAdvisoryAgent — research-only conflict-risk advisory pipeline.

Research simulation only — not for operational use.

Wires the full pipeline:
    ingest (parquet / DataFrame)
    → extract prediction origin groups
    → predict per-aircraft future trajectories
    → compute pairwise closest-point-of-approach (CPA)
    → classify predicted loss-of-separation
    → return ScenarioAdvisoryReport

The agent is stateless between calls: ``advise_scenario()`` is a pure
function of the predictor and the input DataFrame.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.agent.types import (
    ConflictAdvisory,
    PairwiseCPA,
    ScenarioAdvisoryReport,
)
from src.conflict.risk import compute_conflict_risk
from src.data.windows import extract_scenario_origin_groups

logger = logging.getLogger(__name__)


@runtime_checkable
class TrajectoryPredictor(Protocol):
    """Structural protocol for trajectory predictor objects.

    Any object implementing ``predict(history: np.ndarray) -> np.ndarray``
    satisfies this protocol — CV baselines, LSTM, or future models.
    """

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict future trajectory from observation history.

        Parameters
        ----------
        history : np.ndarray
            Shape ``(steps, 3)`` or ``(batch, steps, 3)`` with columns
            ``[x_nm, y_nm, alt_ft]``.

        Returns
        -------
        np.ndarray
            Predicted future positions of shape ``(horizon_steps, 3)``
            or ``(batch, horizon_steps, 3)``.
        """
        ...


class ConflictAdvisoryAgent:
    """Research-only conflict-risk advisory agent.

    Ingests a scenario (as a DataFrame or parquet file), extracts
    prediction origins, runs the configured predictor over every aircraft,
    computes pairwise CPA geometry, classifies predicted loss-of-separation,
    and returns a structured ``ScenarioAdvisoryReport``.

    Parameters
    ----------
    predictor : TrajectoryPredictor
        Trajectory prediction model. Must implement
        ``predict(history: np.ndarray) -> np.ndarray``.
    lateral_min_nm : float, default 5.0
        Lateral separation minimum in nautical miles (ICAO en-route standard).
    vertical_min_ft : float, default 1000.0
        Vertical separation minimum in feet (ICAO en-route standard).
    dt_s : float, default 5.0
        Trajectory resampling interval in seconds.
    horizon_steps : int, default 60
        Number of future prediction steps (60 × 5 s = 300 s look-ahead).
    history_steps : int, default 13
        Number of history steps consumed by the predictor (13 × 5 s = 60 s).
    origin_step_s : float, default 30.0
        Spacing between consecutive prediction origins in seconds.
    use_noisy_history : bool, default True
        When True, use observed (noisy) columns ``x_obs_nm`` / ``y_obs_nm`` /
        ``alt_obs_ft`` for history if present.

    Examples
    --------
    >>> from src.models.baseline import SmoothedConstantVelocityPredictor
    >>> from src.agent import ConflictAdvisoryAgent
    >>> predictor = SmoothedConstantVelocityPredictor()
    >>> agent = ConflictAdvisoryAgent(predictor)
    >>> report = agent.advise_file(Path("data/synthetic/scenario_0.parquet"))
    >>> print(report.has_conflict, report.max_p_risk)
    """

    def __init__(
        self,
        predictor: TrajectoryPredictor,
        lateral_min_nm: float = 5.0,
        vertical_min_ft: float = 1000.0,
        dt_s: float = 5.0,
        horizon_steps: int = 60,
        history_steps: int = 13,
        origin_step_s: float = 30.0,
        use_noisy_history: bool = True,
    ) -> None:
        if not isinstance(predictor, TrajectoryPredictor):
            raise TypeError(
                f"predictor must implement TrajectoryPredictor protocol, got {type(predictor)}"
            )
        if lateral_min_nm <= 0:
            raise ValueError(f"lateral_min_nm must be positive, got {lateral_min_nm}")
        if vertical_min_ft <= 0:
            raise ValueError(f"vertical_min_ft must be positive, got {vertical_min_ft}")
        if dt_s <= 0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be positive, got {horizon_steps}")
        if history_steps < 3:
            raise ValueError(f"history_steps must be >= 3, got {history_steps}")
        if origin_step_s <= 0:
            raise ValueError(f"origin_step_s must be positive, got {origin_step_s}")

        self.predictor = predictor
        self.lateral_min_nm = lateral_min_nm
        self.vertical_min_ft = vertical_min_ft
        self.dt_s = dt_s
        self.horizon_steps = horizon_steps
        self.history_steps = history_steps
        self.origin_step_s = origin_step_s
        self.use_noisy_history = use_noisy_history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise_scenario(self, scenario_df: pd.DataFrame) -> ScenarioAdvisoryReport:
        """Run the full advisory pipeline on an in-memory scenario DataFrame.

        Parameters
        ----------
        scenario_df : pd.DataFrame
            Must contain columns: ``aircraft_id``, ``t``, ``x_nm``, ``y_nm``,
            ``alt_ft``. Optionally ``x_obs_nm``, ``y_obs_nm``, ``alt_obs_ft``
            for noisy history. ``scenario_id`` column is optional but
            recommended.

        Returns
        -------
        ScenarioAdvisoryReport
            Aggregated advisory report for this scenario.

        Raises
        ------
        ValueError
            If required columns are missing or the scenario has fewer than
            2 aircraft.
        """
        required = {"aircraft_id", "t", "x_nm", "y_nm", "alt_ft"}
        missing = required - set(scenario_df.columns)
        if missing:
            raise ValueError(f"scenario_df is missing required columns: {missing}")

        aircraft_ids = sorted(scenario_df["aircraft_id"].unique())
        if len(aircraft_ids) < 2:
            raise ValueError(
                f"advise_scenario requires at least 2 aircraft, got {len(aircraft_ids)}"
            )

        scenario_id: str = (
            str(scenario_df["scenario_id"].iloc[0])
            if "scenario_id" in scenario_df.columns
            else "unknown"
        )

        history_s = (self.history_steps - 1) * self.dt_s  # 60 s
        horizon_s = self.horizon_steps * self.dt_s         # 300 s

        origin_groups = extract_scenario_origin_groups(
            scenario_df,
            origin_step_s=self.origin_step_s,
            history_s=history_s,
            horizon_s=horizon_s,
            dt_s=self.dt_s,
            use_noisy_history=self.use_noisy_history,
        )

        if not origin_groups:
            logger.debug(
                "No valid origin groups extracted for scenario %s — "
                "scenario may be too short for history+horizon.",
                scenario_id,
            )
            return ScenarioAdvisoryReport(
                scenario_id=scenario_id,
                num_aircraft=len(aircraft_ids),
                num_origins=0,
                advisories=(),
                flagged_pairs=frozenset(),
                max_p_risk=0.0,
                earliest_lead_time_s=None,
            )

        all_advisories: list[ConflictAdvisory] = []

        for group in origin_groups:
            # Predict each aircraft's future trajectory from its history
            predictions: dict[str, np.ndarray] = {}
            for acid in group.aircraft_ids:
                hist = group.history[acid]  # (13, 3)
                try:
                    pred = self.predictor.predict(hist)  # (60, 3)
                except Exception:
                    logger.warning(
                        "Predictor failed for aircraft %s at origin %.1f s — skipping.",
                        acid,
                        group.origin_t,
                        exc_info=True,
                    )
                    continue
                predictions[acid] = np.asarray(pred, dtype=np.float64)

            valid_acids = list(predictions.keys())
            if len(valid_acids) < 2:
                continue

            # Evaluate all unique unordered pairs
            for acid_1, acid_2 in itertools.combinations(valid_acids, 2):
                pred_1 = predictions[acid_1]  # (H, 3)
                pred_2 = predictions[acid_2]  # (H, 3)

                cpa = self._compute_cpa(
                    pred_1=pred_1,
                    pred_2=pred_2,
                    origin_t=group.origin_t,
                    scenario_id=scenario_id,
                    aircraft_1=acid_1,
                    aircraft_2=acid_2,
                )
                is_conflict, p_risk = self._classify(cpa)

                # Lead time: origin → first predicted LoS step (if any)
                lead_time_s: float | None = None
                if is_conflict:
                    lead_time_s = self._compute_lead_time(
                        pred_1=pred_1,
                        pred_2=pred_2,
                        origin_t=group.origin_t,
                    )

                all_advisories.append(
                    ConflictAdvisory(
                        cpa=cpa,
                        is_conflict=is_conflict,
                        p_risk=p_risk,
                        lead_time_s=lead_time_s,
                    )
                )

        # Aggregate report
        flagged_pairs: set[tuple[str, str]] = set()
        lead_times: list[float] = []
        max_p_risk = 0.0

        for adv in all_advisories:
            if adv.p_risk > max_p_risk:
                max_p_risk = adv.p_risk
            if adv.is_conflict:
                pair = (adv.cpa.aircraft_1, adv.cpa.aircraft_2)
                flagged_pairs.add(pair)
                if adv.lead_time_s is not None:
                    lead_times.append(adv.lead_time_s)

        earliest_lead_time_s: float | None = min(lead_times) if lead_times else None

        return ScenarioAdvisoryReport(
            scenario_id=scenario_id,
            num_aircraft=len(aircraft_ids),
            num_origins=len(origin_groups),
            advisories=tuple(all_advisories),
            flagged_pairs=frozenset(flagged_pairs),
            max_p_risk=max_p_risk,
            earliest_lead_time_s=earliest_lead_time_s,
        )

    def advise_file(self, path: str | Path, **kwargs: object) -> ScenarioAdvisoryReport:
        """Load a scenario parquet file and run the advisory pipeline.

        Parameters
        ----------
        path : str or Path
            Path to a ``.parquet`` scenario file.
        **kwargs
            Additional keyword arguments forwarded to ``advise_scenario``.

        Returns
        -------
        ScenarioAdvisoryReport

        Raises
        ------
        FileNotFoundError
            If the parquet file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario parquet not found: {path}")
        scenario_df = pd.read_parquet(path)
        return self.advise_scenario(scenario_df, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cpa(
        self,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        origin_t: float,
        scenario_id: str,
        aircraft_1: str,
        aircraft_2: str,
    ) -> PairwiseCPA:
        """Compute closest-point-of-approach from predicted trajectories.

        Parameters
        ----------
        pred_1 : np.ndarray
            Predicted future positions for aircraft 1, shape ``(H, 3)``.
        pred_2 : np.ndarray
            Predicted future positions for aircraft 2, shape ``(H, 3)``.
        origin_t : float
            Prediction origin time (s).
        scenario_id, aircraft_1, aircraft_2 : str
            Metadata fields for the resulting ``PairwiseCPA``.

        Returns
        -------
        PairwiseCPA
        """
        dx = pred_1[:, 0] - pred_2[:, 0]  # (H,)
        dy = pred_1[:, 1] - pred_2[:, 1]  # (H,)
        dz = np.abs(pred_1[:, 2] - pred_2[:, 2])  # (H,)
        lateral_dist = np.hypot(dx, dy)  # (H,)

        cpa_idx = int(np.argmin(lateral_dist))
        cpa_lateral = float(lateral_dist[cpa_idx])
        cpa_vertical = float(dz[cpa_idx])

        # Absolute time at CPA: origin + (step_index + 1) * dt
        cpa_time_s = origin_t + (cpa_idx + 1) * self.dt_s
        time_to_cpa_s = cpa_time_s - origin_t

        # Closure rate: rate of change of lateral distance at origin
        # Use central difference if possible, else forward difference
        if len(lateral_dist) >= 2:
            closure_rate = -(lateral_dist[1] - lateral_dist[0]) / self.dt_s
        else:
            closure_rate = 0.0

        return PairwiseCPA(
            scenario_id=scenario_id,
            aircraft_1=aircraft_1,
            aircraft_2=aircraft_2,
            origin_t=origin_t,
            cpa_lateral_nm=cpa_lateral,
            cpa_vertical_ft=cpa_vertical,
            cpa_time_s=cpa_time_s,
            time_to_cpa_s=time_to_cpa_s,
            closure_rate_nm_per_s=closure_rate,
        )

    def _classify(self, cpa: PairwiseCPA) -> tuple[bool, float]:
        """Classify a CPA as a conflict and compute its risk score.

        A conflict requires **both** lateral and vertical minima to be
        simultaneously violated in the predicted trajectory — matching ICAO
        en-route LoS definition.

        ``p_risk`` is computed as::

            lat_risk  = max(0, 1 - cpa_lateral_nm  / lateral_min_nm)
            vert_risk = max(0, 1 - cpa_vertical_ft / vertical_min_ft)
            p_risk    = min(lat_risk, vert_risk)

        This gives p_risk = 1 at zero separation and 0 when either axis
        is at or beyond its minimum — transparent and non-operational.

        Parameters
        ----------
        cpa : PairwiseCPA

        Returns
        -------
        tuple[bool, float]
            ``(is_conflict, p_risk)``
        """
        is_lateral_violation = cpa.cpa_lateral_nm < self.lateral_min_nm
        is_vertical_violation = cpa.cpa_vertical_ft < self.vertical_min_ft
        is_conflict = is_lateral_violation and is_vertical_violation

        p_risk = compute_conflict_risk(
            cpa_lateral_nm=cpa.cpa_lateral_nm,
            cpa_vertical_ft=cpa.cpa_vertical_ft,
            lateral_min_nm=self.lateral_min_nm,
            vertical_min_ft=self.vertical_min_ft,
        )

        return is_conflict, p_risk

    def _compute_lead_time(
        self,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        origin_t: float,
    ) -> float | None:
        """Return seconds from origin to the first predicted LoS step.

        Parameters
        ----------
        pred_1, pred_2 : np.ndarray
            Predicted trajectories, each shape ``(H, 3)``.
        origin_t : float
            Prediction origin time in seconds.

        Returns
        -------
        float or None
            Lead time in seconds, or None if no LoS step found.
        """
        dx = pred_1[:, 0] - pred_2[:, 0]
        dy = pred_1[:, 1] - pred_2[:, 1]
        dz = np.abs(pred_1[:, 2] - pred_2[:, 2])
        lateral_dist = np.hypot(dx, dy)

        los_mask = (lateral_dist < self.lateral_min_nm) & (dz < self.vertical_min_ft)
        if not np.any(los_mask):
            return None

        first_los_idx = int(np.argmax(los_mask))
        # Step index is 0-based; absolute time = origin + (idx+1)*dt
        first_los_time = origin_t + (first_los_idx + 1) * self.dt_s
        return first_los_time - origin_t
