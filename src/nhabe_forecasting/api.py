"""FastAPI wrapper around the frozen, audited multimodal inference engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from .dashboard import (
    dashboard_html,
    prediction_metadata,
    render_png,
    serialize_pair,
    visualization_array,
)
from .inference import MultimodalInferenceEngine


@dataclass(frozen=True)
class APISettings:
    """Explicit service paths; observations and model artifacts remain external mounts."""

    data_root: Path
    manifest_path: Path
    normalization_path: Path
    model_config_path: Path
    checkpoint_path: Path
    output_root: Path
    device: str = "cpu"

    @classmethod
    def from_environment(cls) -> "APISettings":
        required = {
            "data_root": "NHABE_DATA_ROOT",
            "manifest_path": "NHABE_FORECAST_MANIFEST",
            "normalization_path": "NHABE_NORMALIZATION",
            "model_config_path": "NHABE_MODEL_CONFIG",
            "checkpoint_path": "NHABE_CHECKPOINT",
            "output_root": "NHABE_INFERENCE_OUTPUT",
        }
        missing = [name for name in required.values() if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing API environment variables: {', '.join(missing)}")
        return cls(
            **{field: Path(os.environ[name]).resolve() for field, name in required.items()},
            device=os.environ.get("NHABE_DEVICE", "cpu"),
        )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str = Field(min_length=1, max_length=180, pattern=r"^[A-Za-z0-9_.-]+$")


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    artifact_id: str
    pair_id: str
    input_time_utc: str
    forecast_valid_time_utc: str
    lead_time_minutes: int
    valid_pixels: int
    valid_percent: float
    runtime_seconds: float
    unit: str
    npz_artifact: str
    json_artifact: str
    npz_sha256: str
    future_target_array_read: bool


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Create an application; heavy model initialization occurs in its lifespan."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        resolved = settings or APISettings.from_environment()
        resolved.output_root.mkdir(parents=True, exist_ok=True)
        engine = MultimodalInferenceEngine(
            data_root=resolved.data_root,
            manifest_path=resolved.manifest_path,
            normalization_path=resolved.normalization_path,
            model_config_path=resolved.model_config_path,
            checkpoint_path=resolved.checkpoint_path,
            requested_device=resolved.device,
        )
        application.state.settings = resolved
        application.state.engine = engine
        application.state.inference_lock = threading.Lock()
        yield

    application = FastAPI(
        title="Nha Be Radar–Himawari Forecast API",
        version="1.1.0",
        description=(
            "Offline research API for one audited ten-minute reflectivity forecast. "
            "It is not an operational weather service."
        ),
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/demo")

    @application.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return dashboard_html()

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        info = request.app.state.engine.model_info()
        return {
            "status": "READY",
            "model_loaded": True,
            "device": info["device"],
            "output_schema": info["output_schema"],
        }

    @application.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        return request.app.state.engine.model_info()

    @application.get("/pairs")
    def pairs(
        request: Request,
        split: Literal["train", "validation", "test"] = "validation",
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        engine: MultimodalInferenceEngine = request.app.state.engine
        rows = engine.available_pair_rows(split, limit)
        return {"split": split, "count": len(rows), "pairs": [serialize_pair(row) for row in rows]}

    @application.get("/predictions/{pair_id}")
    def existing_prediction(pair_id: str, request: Request) -> dict[str, Any]:
        engine: MultimodalInferenceEngine = request.app.state.engine
        try:
            engine.pair_row(pair_id)
            return prediction_metadata(request.app.state.settings.output_root, pair_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="pair_id is not in the audited manifest") from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="prediction artifact does not exist") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/visualization/{pair_id}/{layer}.png", response_class=Response)
    def visualization(
        pair_id: str,
        layer: Literal["current", "prediction", "target", "error"],
        request: Request,
    ) -> Response:
        engine: MultimodalInferenceEngine = request.app.state.engine
        try:
            row = engine.pair_row(pair_id)
            values, mask, vmin, vmax, stops = visualization_array(
                data_root=request.app.state.settings.data_root,
                output_root=request.app.state.settings.output_root,
                row=row,
                layer=layer,
            )
            image = render_png(values, mask, vmin, vmax, stops)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="pair_id is not in the audited manifest") from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return Response(
            content=image,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-NHABE-Historical-Target-Read": str(layer in {"target", "error"}).lower(),
            },
        )

    @application.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        output_root: Path = request.app.state.settings.output_root
        pair_output = output_root / payload.pair_id
        try:
            with request.app.state.inference_lock:
                result = request.app.state.engine.predict(payload.pair_id, pair_output)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="pair_id is not in the audited manifest") from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail="prediction artifact already exists") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return PredictResponse(
            status=result["status"], artifact_id=payload.pair_id, pair_id=payload.pair_id,
            input_time_utc=result["input_time_utc"],
            forecast_valid_time_utc=result["forecast_valid_time_utc"],
            lead_time_minutes=int(result["lead_time_minutes"]),
            valid_pixels=int(result["valid_pixels"]),
            valid_percent=float(result["valid_percent"]),
            runtime_seconds=float(result["runtime_seconds"]), unit=result["unit"],
            npz_artifact=f"{payload.pair_id}/{payload.pair_id}_prediction.npz",
            json_artifact=f"{payload.pair_id}/{payload.pair_id}_prediction.json",
            npz_sha256=result["npz_sha256"],
            future_target_array_read=bool(result["future_target_array_read"]),
        )

    return application


app = create_app()
