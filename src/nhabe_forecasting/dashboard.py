"""Self-contained local dashboard helpers for audited forecast products.

The dashboard is deliberately thin: inference still goes through the FastAPI
``/predict`` contract. Historical target data is read only when a user explicitly
requests an evaluation layer.
"""

from __future__ import annotations

import binascii
from pathlib import Path
import struct
from typing import Any, Literal, Mapping
import zlib

import numpy as np

from .inference import validate_prediction_files
from .normalization import valid_values


LayerName = Literal["current", "prediction", "target", "error"]

REFLECTIVITY_STOPS = np.asarray([
    [-10.0, 49, 18, 59],
    [0.0, 65, 105, 225],
    [10.0, 30, 190, 225],
    [20.0, 50, 240, 130],
    [30.0, 210, 230, 45],
    [40.0, 255, 160, 30],
    [50.0, 235, 60, 20],
    [60.0, 122, 4, 3],
], dtype=np.float64)

ERROR_STOPS = np.asarray([
    [0.0, 0, 0, 4],
    [5.0, 55, 15, 105],
    [10.0, 125, 30, 135],
    [20.0, 230, 85, 75],
    [30.0, 252, 245, 190],
], dtype=np.float64)


def _source_path(data_root: Path, value: str) -> Path:
    root = data_root.resolve()
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Visualization source path escapes the data root")
    return resolved


def _read_radar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays["radar_reflectivity_dbz"].copy()
        mask = arrays["radar_data_valid_mask"].copy()
    valid_values(values, mask, (256, 256))
    return values, mask


def _read_prediction(
    output_root: Path, pair_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pair_root = output_root.resolve() / pair_id
    npz_path = pair_root / f"{pair_id}_prediction.npz"
    json_path = pair_root / f"{pair_id}_prediction.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError("Prediction artifact does not exist")
    metadata = validate_prediction_files(npz_path, json_path, pair_id)
    with np.load(npz_path, allow_pickle=False) as arrays:
        values = arrays["predicted_reflectivity_dbz"].copy()
        mask = arrays["prediction_valid_mask"].copy()
    return values, mask, metadata


def prediction_metadata(output_root: Path, pair_id: str) -> dict[str, Any]:
    """Return validated metadata without exposing a local filesystem path."""
    _values, _mask, metadata = _read_prediction(output_root, pair_id)
    allowed = (
        "status", "pair_id", "split", "input_time_utc", "forecast_valid_time_utc",
        "lead_time_minutes", "input_mode", "checkpoint_epoch", "device",
        "runtime_seconds", "unit", "valid_pixels", "valid_percent", "npz_sha256",
        "future_target_array_read",
    )
    return {name: metadata[name] for name in allowed}


def visualization_array(
    *, data_root: Path, output_root: Path, row: Mapping[str, str], layer: LayerName,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Load one display layer; target data is touched only for target/error layers."""
    pair_id = row["pair_id"]
    if layer == "current":
        values, mask = _read_radar(_source_path(data_root, row["input_npz_path"]))
        return values, mask, -10.0, 60.0, REFLECTIVITY_STOPS
    if layer == "prediction":
        values, mask, _metadata = _read_prediction(output_root, pair_id)
        return values, mask, -10.0, 60.0, REFLECTIVITY_STOPS
    if layer == "target":
        values, mask = _read_radar(_source_path(data_root, row["target_npz_path"]))
        return values, mask, -10.0, 60.0, REFLECTIVITY_STOPS
    if layer == "error":
        prediction, prediction_mask, _metadata = _read_prediction(output_root, pair_id)
        target, target_mask = _read_radar(_source_path(data_root, row["target_npz_path"]))
        mask = prediction_mask & target_mask
        values = np.full((256, 256), np.nan, dtype=np.float32)
        values[mask] = np.abs(prediction[mask] - target[mask])
        return values, mask, 0.0, 30.0, ERROR_STOPS
    raise ValueError(f"Unsupported visualization layer: {layer}")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload)) + kind + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def render_png(
    values: np.ndarray, mask: np.ndarray, vmin: float, vmax: float, stops: np.ndarray,
) -> bytes:
    """Render one 256x256 masked field as a dependency-free RGBA PNG."""
    if values.shape != (256, 256) or mask.shape != values.shape:
        raise ValueError("Visualization arrays must have shape 256x256")
    if values.dtype != np.float32 or mask.dtype != np.bool_:
        raise ValueError("Visualization requires float32 values and a boolean mask")
    if not np.array_equal(np.isfinite(values), mask):
        raise ValueError("Visualization NaN pattern differs from its boolean mask")
    clipped = np.where(
        mask,
        np.clip(values.astype(np.float64, copy=False), vmin, vmax),
        vmin,
    )
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    for channel in range(3):
        rgba[..., channel] = np.interp(
            clipped, stops[:, 0], stops[:, channel + 1]
        ).astype(np.uint8)
    rgba[..., 3] = 255
    rgba[~mask] = np.asarray([242, 244, 247, 255], dtype=np.uint8)
    # Radar arrays use row zero at the southern edge; PNG row zero is displayed at the top.
    displayed = np.ascontiguousarray(rgba[::-1])
    raw = b"".join(b"\x00" + displayed[row].tobytes() for row in range(256))
    header = struct.pack(">IIBBBBB", 256, 256, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nha Be Radar–Himawari Forecast Demo</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#5f6b76; --line:#d9e0e6;
      --blue:#1769aa; --navy:#0d2742; --surface:#ffffff; --bg:#f4f7f9; --good:#147d64; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,sans-serif; }
    header { background:linear-gradient(120deg,var(--navy),#185f8f); color:white; padding:28px max(20px,calc((100% - 1240px)/2)); }
    header h1 { margin:0 0 6px; font-size:clamp(24px,4vw,38px); }
    header p { margin:0; max-width:900px; color:#dcecf7; }
    main { max-width:1240px; margin:24px auto; padding:0 20px 48px; }
    .notice { border-left:4px solid #e09f3e; background:#fff8e8; padding:12px 15px; margin-bottom:18px; }
    .panel { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px; margin-bottom:18px; box-shadow:0 3px 14px #17324d12; }
    .status-grid,.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
    .metric { background:#f7fafc; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
    .metric span { display:block; color:var(--muted); font-size:12px; }
    .metric strong { display:block; overflow-wrap:anywhere; }
    label { font-weight:650; }
    select,button { width:100%; padding:11px 12px; border-radius:8px; border:1px solid #aeb9c3; font:inherit; }
    button { border:0; background:var(--blue); color:white; font-weight:700; cursor:pointer; margin-top:10px; }
    button:disabled { background:#91a4b4; cursor:wait; }
    .toggle { display:flex; gap:9px; align-items:flex-start; margin-top:12px; font-weight:400; }
    .toggle input { margin-top:4px; }
    .message { min-height:24px; margin:10px 0 0; font-weight:650; }
    .good { color:var(--good); } .bad { color:#b42318; }
    .plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .plot { border:1px solid var(--line); border-radius:10px; padding:12px; background:white; }
    .plot h3 { margin:0 0 8px; }
    .plot img { width:100%; aspect-ratio:1; object-fit:contain; background:#f2f4f7; image-rendering:auto; }
    .bar { height:12px; border-radius:6px; margin-top:9px; }
    .refl { background:linear-gradient(90deg,#31123b,#4169e1,#1ebee1,#32f082,#d2e62d,#ffa01e,#eb3c14,#7a0403); }
    .err { background:linear-gradient(90deg,#000004,#370f69,#7d1e87,#e6554b,#fcf5be); }
    .ticks { display:flex; justify-content:space-between; color:var(--muted); font-size:11px; }
    .hidden { display:none; }
    code { background:#edf2f6; border-radius:4px; padding:2px 4px; overflow-wrap:anywhere; }
    footer { color:var(--muted); margin-top:20px; }
    @media (max-width:720px) { .plots { grid-template-columns:1fr; } header { padding:22px 18px; } main { padding:0 12px 36px; } }
  </style>
</head>
<body>
<header>
  <h1>Nha Be Radar–Himawari Forecast Demo</h1>
  <p>Interactive visualization of one audited historical reflectivity forecast.</p>
</header>
<main>
  <div id="notice" class="notice"><strong>Research demonstration.</strong> Not an operational weather service. Display ranges are visualization choices, not processing thresholds.</div>
  <section class="panel">
    <h2>1. Service and model</h2>
    <div id="status" class="status-grid"><div class="metric"><span>Status</span><strong>Loading…</strong></div></div>
  </section>
  <section class="panel">
    <h2>2. Select an audited pair</h2>
    <label for="pair">Validation pair</label>
    <select id="pair"></select>
    <button id="predict">Run forecast</button>
    <label class="toggle"><input id="evaluation" type="checkbox"> <span>Historical evaluation view — explicitly read the future target after inference to display observation and absolute error.</span></label>
    <p id="message" class="message"></p>
  </section>
  <section id="results" class="panel hidden">
    <h2>3. Forecast result</h2>
    <div id="metrics" class="metric-grid"></div>
    <div class="plots" style="margin-top:16px">
      <article class="plot"><h3>Current radar</h3><img id="current" alt="Current radar reflectivity"><div class="bar refl"></div><div class="ticks"><span>−10</span><span>Reflectivity (dBZ)</span><span>60</span></div></article>
      <article class="plot"><h3>+10 min prediction</h3><img id="prediction" alt="Predicted radar reflectivity"><div class="bar refl"></div><div class="ticks"><span>−10</span><span>Reflectivity (dBZ)</span><span>60</span></div></article>
      <article id="target-card" class="plot hidden"><h3>+10 min observation</h3><img id="target" alt="Future observed radar reflectivity"><div class="bar refl"></div><div class="ticks"><span>−10</span><span>Reflectivity (dBZ)</span><span>60</span></div></article>
      <article id="error-card" class="plot hidden"><h3>Absolute error</h3><img id="error" alt="Absolute prediction error"><div class="bar err"></div><div class="ticks"><span>0</span><span>Absolute error (dBZ)</span><span>30</span></div></article>
    </div>
  </section>
  <footer>API documentation: <a href="/docs">Swagger UI</a>. Inference never opens the future target; the optional evaluation images are requested separately.</footer>
</main>
<script>
const byId = id => document.getElementById(id);
const escapeHtml = value => String(value ?? "—").replace(/[&<>"']/g, character => ({
  '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;'
})[character]);
async function jsonFetch(url, options) {
  const response = await fetch(url, options);
  let body = {}; try { body = await response.json(); } catch (_) {}
  if (!response.ok) { const error = new Error(body.detail || `HTTP ${response.status}`); error.status = response.status; throw error; }
  return body;
}
function metric(label, value) { return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`; }
async function loadPage() {
  try {
    const [health, model, pairs] = await Promise.all([
      jsonFetch('/health'), jsonFetch('/model-info'), jsonFetch('/pairs?split=validation&limit=50')
    ]);
    byId('status').innerHTML = metric('Service', health.status) + metric('Model', model.input_mode) + metric('Checkpoint epoch', model.checkpoint_epoch) + metric('Trained checkpoint', model.trained_checkpoint) + metric('Artifact kind', model.artifact_kind) + metric('Device', model.device) + metric('Audited pairs', model.available_pairs);
    if (!model.trained_checkpoint) byId('notice').innerHTML = '<strong>Synthetic portfolio demo.</strong> Inputs are generated arrays and the checkpoint has untrained random weights. These images demonstrate the software contract only, not forecast skill.';
    byId('pair').innerHTML = pairs.pairs.map(item => `<option value="${escapeHtml(item.pair_id)}">${escapeHtml(item.input_time_utc)} → ${escapeHtml(item.target_time_utc)}</option>`).join('');
  } catch (error) { byId('message').textContent = error.message; byId('message').className = 'message bad'; }
}
function showImages(pairId, evaluation) {
  const stamp = Date.now();
  byId('current').src = `/visualization/${encodeURIComponent(pairId)}/current.png?t=${stamp}`;
  byId('prediction').src = `/visualization/${encodeURIComponent(pairId)}/prediction.png?t=${stamp}`;
  for (const name of ['target','error']) {
    byId(`${name}-card`).classList.toggle('hidden', !evaluation);
    if (evaluation) byId(name).src = `/visualization/${encodeURIComponent(pairId)}/${name}.png?t=${stamp}`;
    else byId(name).removeAttribute('src');
  }
}
async function runForecast() {
  const pairId = byId('pair').value; if (!pairId) return;
  const button = byId('predict'); button.disabled = true; button.textContent = 'Running…';
  byId('message').textContent = 'Loading inputs and running the frozen model…'; byId('message').className = 'message';
  try {
    let result;
    try { result = await jsonFetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({pair_id:pairId})}); }
    catch (error) { if (error.status !== 409) throw error; result = await jsonFetch(`/predictions/${encodeURIComponent(pairId)}`); }
    byId('metrics').innerHTML = metric('Input time UTC', result.input_time_utc) + metric('Forecast-valid UTC', result.forecast_valid_time_utc) + metric('Runtime', `${Number(result.runtime_seconds).toFixed(3)} s`) + metric('Valid pixels', `${result.valid_pixels} (${Number(result.valid_percent).toFixed(2)}%)`) + metric('Unit', result.unit) + metric('Future target read by inference', result.future_target_array_read);
    showImages(pairId, byId('evaluation').checked); byId('results').classList.remove('hidden');
    byId('message').textContent = result.status === 'PASS' ? 'Forecast PASS. Existing audited output is safely reused when present.' : result.status;
    byId('message').className = 'message good';
  } catch (error) { byId('message').textContent = error.message; byId('message').className = 'message bad'; }
  finally { button.disabled = false; button.textContent = 'Run forecast'; }
}
byId('predict').addEventListener('click', runForecast);
byId('evaluation').addEventListener('change', () => {
  if (!byId('results').classList.contains('hidden')) showImages(byId('pair').value, byId('evaluation').checked);
});
loadPage();
</script>
</body>
</html>
"""


def dashboard_html() -> str:
    """Return the static application shell without embedding data or local paths."""
    return DASHBOARD_HTML


def serialize_pair(row: Mapping[str, str]) -> dict[str, str]:
    """Expose only identifiers and times required by the local demo selector."""
    return {
        "pair_id": row["pair_id"],
        "split": row["split"],
        "input_time_utc": row["input_time_utc"],
        "target_time_utc": row["target_time_utc"],
    }
