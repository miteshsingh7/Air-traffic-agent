"""FastAPI web application for AeroMetrics Conflict Research Console.

Research simulation only - not for operational use.
Serves REST API endpoints and static research console frontend.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.ui.service import (
    get_benchmark_metrics,
    get_system_info,
    list_datasets,
    list_scenarios,
    load_scenario_detail,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Pre-verify system info on startup
    _ = get_system_info()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AeroMetrics Conflict Research Console",
        description="Air-Traffic Conflict-Risk Advisory Agent Research Workbench",
        version="2.4.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # wildccard origin + credentials=True is invalid per Fetch spec
        allow_methods=["GET"],    # research console is read-only
        allow_headers=["*"],
    )

    @app.get("/api/system")
    def api_system() -> dict[str, Any]:
        """Return system configuration, device, and dataset parameters."""
        return get_system_info()

    @app.get("/api/datasets")
    def api_datasets() -> list[dict[str, Any]]:
        """List available datasets, splits, and scenario counts."""
        return list_datasets()

    @app.get("/api/scenarios")
    def api_scenarios(
        dataset: str = Query("hard_v1", description="Dataset identifier"),
        split: str = Query("test", description="Dataset split (test/val/train)"),
    ) -> list[dict[str, Any]]:
        """List scenarios in dataset split with conflict labels."""
        try:
            return list_scenarios(dataset=dataset, split=split)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/scenario/{scenario_id}")
    def api_scenario_detail(
        scenario_id: str,
        dataset: str = Query("hard_v1", description="Dataset identifier"),
        split: str = Query("test", description="Dataset split"),
        origin: float | None = Query(None, description="Observation origin time in seconds"),
        model: str = Query("cv_smoothed", description="Trajectory prediction model name"),
        lookahead: float = Query(180.0, description="Predictive horizon lookahead in seconds"),
    ) -> dict[str, Any]:
        """Load scenario trajectory tracks, model prediction, and pairwise conflict telemetry."""
        try:
            return load_scenario_detail(
                dataset=dataset,
                split=split,
                scenario_id=scenario_id,
                origin_t=origin,
                model_name=model,
                lookahead_s=lookahead,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/metrics")
    def api_metrics(
        dataset: str = Query("hard_v1", description="Dataset identifier"),
        split: str = Query("test", description="Dataset split"),
        model: str = Query("cv_smoothed", description="Model name (cv_3step, cv_smoothed, lstm_v1)"),
    ) -> dict[str, Any]:
        """Return verified benchmark evaluation metrics for model and split."""
        try:
            return get_benchmark_metrics(dataset=dataset, split=split, model_name=model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
