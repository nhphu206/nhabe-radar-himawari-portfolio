# Docker build and sample validation

The first ARM64 build failed while compiling Py-ART because gcc was missing. The Dockerfile
now has a separate compiler-equipped builder. The corrected Linux ARM64 build succeeded;
all five unit tests passed inside the container, and the bind-mounted mock output passed
validation again when reopened with the host Python environment.
Use commands from this portfolio repository, not the private research workspace.

```bash
docker build --progress=plain -t nhabe-preprocessing:dev .
```

The base is the official Python 3.12 Debian slim image. Dependency versions match the
existing research environment. Scientific settings in `configs/pipeline.yaml` are unchanged.
The builder installs `build-essential`, creates `/opt/venv`, builds dependencies, and installs
this package. The final runtime stage copies that environment onto the same Python base,
leaving the compiler toolchain behind. It runs the environment check as a non-root user
(UID/GID 10001). The base tag is not yet pinned to a digest and
transitive dependencies are not fully locked.

Py-ART 2.2.5 was selected as a source archive in the observed Linux ARM64/Python 3.12 build.
The recorded failure was `No such file or directory: 'gcc'`, after roughly six minutes in
the dependency step. Source compilation can take several minutes. BuildKit caches pip
archives/wheels for retries without copying that cache into the final image. Further missing
system dependencies must be diagnosed from the retry log; scientific-library versions remain
unchanged.

`.dockerignore` uses a file allowlist. The Dockerfile also copies explicit code, tests,
and configuration paths. No observations, generated arrays, Git history, reports,
internal handoff notes, or credentials should enter the build context or image.

Environment and contract checks:

```bash
docker run --rm nhabe-preprocessing:dev doctor
docker run --rm nhabe-preprocessing:dev test
docker run --rm nhabe-preprocessing:dev sample
docker image inspect nhabe-preprocessing:dev --format '{{.Size}}'
```

Without an output mount, the sample remains inside the container and is discarded when
`--rm` removes it. The following persistent mock-output run has been verified on Docker Desktop
for macOS ARM64:

```bash
mkdir -p output/docker-sample
docker run --rm \
  --mount type=bind,source="$(pwd)/output/docker-sample",target=/app/output \
  nhabe-preprocessing:dev sample --output /app/output
```

The synthetic NPZ/JSON sample was reopened on the host: radar shape `[256,256]`, satellite
shape `[5,256,256]`, both physical arrays float32, and finite-value/boolean-mask relationships
valid. This verifies the sample contract and output persistence; it does not verify SIGMET
reading, HSD calibration, or real-data interpolation inside Docker.

The verified image manifest-list ID begins `a6ac314e0ec5`. Docker Desktop's image listing
reported **1.22 GB DISK USAGE** and **271 MB CONTENT SIZE**. These are the two distinct values
reported by that Docker installation, not the RAM requirement or dataset size. Image sizes
can change across architectures, base-image updates, and dependency resolution.

Compose mock execution and read-only output revalidation passed on Docker Desktop; see
[the walkthrough](compose.md). Real-input processing remains to be verified.
Bind-mounted output directories must allow writes by the container user. Real-data workflows
also need read-only configuration/raw-input mounts, writable staging and output locations,
and previously prepared timing inputs. No real-data month run is triggered by these commands.

The entry point is `nhabe-pipeline`; Docker appends command arguments to it. For example,
`docker run --rm nhabe-preprocessing:dev doctor` runs the environment check. With no arguments,
the container prints CLI help. A continuously running health check is not used for this batch
application; `doctor` is the explicit environment check.
