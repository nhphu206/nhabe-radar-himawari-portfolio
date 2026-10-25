# Forecast inference API

The FastAPI service is a local, offline research wrapper around the frozen multimodal model.
It accepts only a `pair_id` already present in the audited forecast manifest. Clients cannot
submit filesystem paths, upload arbitrary arrays or change model/scientific parameters.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /demo` | Open the interactive local forecast dashboard |
| `GET /health` | Confirm the model is loaded and the service is ready |
| `GET /model-info` | Return checkpoint/config hashes, epoch, device and pair count |
| `GET /pairs` | List safe identifiers and timestamps from one audited split |
| `POST /predict` | Generate one audited NPZ/JSON prediction for `{"pair_id": "..."}` |
| `GET /predictions/{pair_id}` | Return validated metadata for an existing prediction |
| `GET /visualization/{pair_id}/{layer}.png` | Render `current`, `prediction`, `target` or `error` as a 256 × 256 PNG |

The API returns relative artifact identifiers, not server filesystem paths. Unknown pairs
return HTTP 404, an existing output returns 409, and invalid or extra request fields return
422. A process lock serializes inference, and Uvicorn runs one worker so the checkpoint has one
in-memory copy. The model is loaded and fingerprinted during application startup.

The browser dashboard is served by the same FastAPI process and adds no frontend framework or
database. Its default inference view reads the current radar/Himawari inputs, runs the frozen
model and displays current reflectivity beside the ten-minute prediction. It does **not** open
the future target. Selecting **Historical evaluation view** makes separate HTTP requests for the
future observation and absolute-error image after inference. The response header
`X-NHABE-Historical-Target-Read` records that distinction for every rendered layer.

## Run locally

Set explicit paths and start the server from the dedicated AI environment:

```bash
./.venv/bin/python -m pip install -r requirements-forecasting.txt
./.venv/bin/python -m pip install -r requirements-api.txt

export NHABE_DATA_ROOT=/path/to/research-workspace
export NHABE_FORECAST_MANIFEST=/path/to/forecast_pairs.csv
export NHABE_NORMALIZATION=/path/to/normalization.json
export NHABE_MODEL_CONFIG="$PWD/configs/model.yaml"
export NHABE_CHECKPOINT=/path/to/multimodal/best_model.pt
export NHABE_INFERENCE_OUTPUT=/new/empty/output-root
export NHABE_DEVICE=cpu

./.venv/bin/uvicorn nhabe_forecasting.api:app \
  --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/demo> for the visual interface or
<http://127.0.0.1:8000/docs> for Swagger UI. Choose an audited validation pair and press
**Run forecast**. If that pair was already generated, the dashboard safely reopens and
validates the existing NPZ/JSON artifacts rather than overwriting them.

## One-command synthetic visual demo

Reviewers without the private observations or trained checkpoint can exercise the complete UI:

```bash
docker compose -f compose.demo.yaml up --build
```

The service is available at <http://127.0.0.1:8000/demo>. The Compose configuration uses only
temporary in-container files. It generates deterministic synthetic radar, Himawari and future
target arrays together with an untrained random-weight checkpoint. `/model-info` exposes
`trained_checkpoint=false` and `artifact_kind=SYNTHETIC_INTERACTIVE_DEMO_FIXTURE`; the dashboard
shows a matching warning. This path demonstrates integration contracts, not meteorological skill.
It neither reads nor mounts the private research dataset.

Example request:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"pair_id":"PAIR_ID"}'
```

## AI Docker image

`Dockerfile.ai` is separate from the preprocessing image. It includes API/model source and the
model configuration, but excludes observations, manifests, normalization artifacts,
checkpoints and generated predictions. It installs the CPU-only PyTorch wheel from PyTorch's
official CPU index, avoiding unused CUDA libraries. Compose mounts data/artifacts/model
read-only and output writable; the port is bound to loopback only.

```bash
mkdir -p output/api
export AI_DATA_ROOT=/path/to/research-workspace
export AI_MANIFEST_FILE=/path/to/forecast_pairs.csv
export AI_NORMALIZATION_FILE=/path/to/normalization.json
export AI_CHECKPOINT_FILE=/path/to/best_model.pt
export AI_OUTPUT_DIR="$PWD/output/api"

docker compose -f compose.ai.yaml config --quiet
docker compose -f compose.ai.yaml up --build
```

Compose mounts each artifact at its expected container filename. A deployment may instead set
the `NHABE_*` container environment variables, but paths must remain explicit. The output
directory must already exist and be writable by UID 10002, or `AI_UID`/`AI_GID` must match its
owner.

## Limitations and security boundary

This API is intentionally not described as production-ready. It has no authentication, TLS,
rate limiting, job queue, artifact download endpoint, metrics backend or operational data
ingestion. Keep it on localhost or a trusted private network. It works from an offline audited
manifest whose rows include a future target identity, although inference never opens the target
array. It does not prove real-time data availability or an operational ten-minute warning lead.

## Verified local and container checks

On October 25, 2026, the local Uvicorn service returned HTTP 200 for `/health`, `/model-info`,
`/demo`, `/pairs`, the four visualization layers and a real validation `/predict` request. The
model-info response matched checkpoint epoch 23,
all frozen fingerprints and 822 audited pairs. The response exposed relative artifact names and
reported `future_target_array_read=false`. The current and prediction layers confirmed that the
inference view did not read the historical target; target and error were loaded only through the
explicit evaluation view.

The ARM64 AI image built successfully with PyTorch `2.14.1+cpu` and no CUDA runtime. Docker
reported 278,803,895 bytes (approximately 266 MiB), and `pip check` reported no broken
requirements. Compose reached `healthy`; a real mounted-data prediction was reopened on the
host with shape 256 × 256, float32 values, 7,452 valid pixels and matching NPZ hash. A static
image scan found no `.pt`, `.pth`, `.npz` or raw-radar file under `/app`. The test container and
network were removed afterward. GitHub Actions run
[40172733715](https://github.com/nhphu206/nhabe-radar-himawari-portfolio/actions/runs/40172733715)
exercised the complete synthetic dashboard/API contract on Linux AMD64: 31 AI/API tests, image
build, dashboard shell, pair listing, current/prediction PNG dimensions, target-free inference,
one-command interactive demo, prediction reopen, image-content scan and cleanup.
