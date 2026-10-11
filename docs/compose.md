# Compose: explicit configuration, input, and output mounts

Compose records Docker run options in YAML. It uses the same Dockerfile/image and scientific
configuration. The synthetic sample run passed on macOS ARM64 with Docker Desktop. The
persisted NPZ/JSON output also passed validation when reopened through a read-only mount.

## Synthetic sample

From the portfolio root, with the previously built `nhabe-preprocessing:dev` image:

```bash
mkdir -p output/compose-sample
docker compose config --quiet
docker compose run --rm pipeline sample --output /app/output
```

`pipeline` is the service name, not a processing stage. Arguments after it are passed to
`nhabe-pipeline`. `config --quiet` validates the Compose file without creating a container.
The run command executes one container and removes it on completion. It does not rebuild the
image; if the image is missing or source code changed, first run `docker compose build`.

| Host source | Container target | Mode |
| --- | --- | --- |
| `./configs` | `/app/configs` | Read-only |
| `./output/compose-sample` | `/app/output` | Read/write |

The container root filesystem is also read-only. `/tmp` is a temporary writable filesystem for
library caches and temporary files. Generated NPZ/JSON files persist only in the output mount.
Raw observations are not needed or mounted for the synthetic workflow.

To reopen the written output through a read-only mount:

```bash
docker compose run --rm --entrypoint python \
  -v "$(pwd)/output/compose-sample:/verified:ro" pipeline -c '
from pathlib import Path
from nhabe_preprocessing.validation import validate_sample_files
root = Path("/verified")
stem = "MOCK_NHB_20250801T000000Z_A"
validate_sample_files(root / f"{stem}.npz", root / f"{stem}.json", "MOCK_NOT_RAW_DATA")
print("Reopened Compose output: PASS")
'
```

Docker Desktop allowed this Compose output pattern with UID/GID 10001. On Linux, prepare an
output directory writable by the container user, or run with the host user's non-root UID/GID:

```bash
PIPELINE_UID=$(id -u) PIPELINE_GID=$(id -g) \
  docker compose run --rm pipeline sample --output /app/output
```

Relative mount paths resolve against the Compose project directory. The same workflow can be
invoked from another directory with `docker compose -f /absolute/path/to/compose.yaml ...`.

## Optional real-data overlay — a separate step

`compose.real-data.yaml` adds two mounts only when explicitly selected. Set `RAW_INPUT_DIR`
to an existing authorized raw-data directory, and `STAGING_DIR` to an existing writable staging
directory. Use a separate `PIPELINE_OUTPUT_DIR` for a limited experiment. Never point output or
staging at raw data or at the audited dataset.

```text
RAW_INPUT_DIR  -> /app/data/raw      read-only
STAGING_DIR    -> /app/data/staging  read/write
```

The input layout must match `configs/pipeline.yaml`, including the radar date directories and
the day-01 Himawari path. Time manifests and header inputs must be prepared before preprocessing.
Later, after verifying those paths and a small sample, use both Compose files explicitly:

```bash
docker compose -f compose.yaml -f compose.real-data.yaml config --quiet
```

Without the required variables, that command fails rather than guessing an input location.
All bind mounts set `create_host_path: false` so missing paths fail at run time instead of being
silently created. The default command only prints CLI help. No real-data run or download is
triggered by configuration validation. Real-data processing through this overlay is unverified.

References: [Compose services](https://docs.docker.com/reference/compose-file/services/),
[Compose run](https://docs.docker.com/reference/cli/docker/compose/run/).
