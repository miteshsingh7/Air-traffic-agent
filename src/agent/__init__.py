"""Agent module for conflict risk advisory and resolution.

Research simulation only — not for operational use.

Public API
----------
ConflictAdvisoryAgent
    Full pipeline: ingest → predict → detect → advise.
PairwiseCPA
    Closest-point-of-approach geometry (predicted).
ConflictAdvisory
    Advisory signal for a single (pair, origin) sample.
ScenarioAdvisoryReport
    Aggregated report for a full scenario.
"""

from src.agent.advisor import ConflictAdvisoryAgent, TrajectoryPredictor
from src.agent.types import ConflictAdvisory, PairwiseCPA, ScenarioAdvisoryReport

__all__ = [
    "ConflictAdvisoryAgent",
    "TrajectoryPredictor",
    "ConflictAdvisory",
    "PairwiseCPA",
    "ScenarioAdvisoryReport",
]
