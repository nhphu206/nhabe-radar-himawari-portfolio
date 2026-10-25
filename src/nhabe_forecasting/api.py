"""FastAPI wrapper around the frozen, audited multimodal inference engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

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
        version="1.0.0",
        description=(
            "Offline research API for one audited ten-minute reflectivity forecast. "
            "It is not an operational weather service."
        ),
        lifespan=lifespan,
    )

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
