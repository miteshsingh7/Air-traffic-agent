"""Conflict risk calculation module.

Research simulation only — not for operational use.
"""

from __future__ import annotations


def compute_conflict_risk(
    cpa_lateral_nm: float,
    cpa_vertical_ft: float,
    lateral_min_nm: float = 5.0,
    vertical_min_ft: float = 1000.0,
) -> float:
    """Compute continuous conflict risk score P(risk) in [0, 1].

    Defined as min(lat_risk, vert_risk), where each axis risk decays
    linearly from 1.0 at zero separation to 0.0 at the separation minimum.

    Parameters
    ----------
    cpa_lateral_nm : float
        Closest predicted lateral separation in nautical miles.
    cpa_vertical_ft : float
        Vertical separation at closest lateral point in feet.
    lateral_min_nm : float, default 5.0
        Lateral separation minimum in nautical miles.
    vertical_min_ft : float, default 1000.0
        Vertical separation minimum in feet.

    Returns
    -------
    float
        Risk score in [0.0, 1.0].
    """
    if lateral_min_nm <= 0:
        raise ValueError(f"lateral_min_nm must be positive, got {lateral_min_nm}")
    if vertical_min_ft <= 0:
        raise ValueError(f"vertical_min_ft must be positive, got {vertical_min_ft}")

    lat_risk = max(0.0, 1.0 - cpa_lateral_nm / lateral_min_nm)
    vert_risk = max(0.0, 1.0 - cpa_vertical_ft / vertical_min_ft)
    return float(min(lat_risk, vert_risk))
