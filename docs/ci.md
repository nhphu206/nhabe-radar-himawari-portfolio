# Continuous integration

The workflow in `.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch.
The first hosted run passed both jobs, as confirmed by the provided logs and screenshot:
[run #1](https://github.com/nhphu206/nhabe-radar-himawari-portfolio/actions/runs/38146073831).
Python took 1 minute 2 seconds; Docker checks took 1 minute 28 seconds.
The expanded test suite subsequently passed both jobs in
[run #3](https://github.com/nhphu206/nhabe-radar-himawari-portfolio/actions/runs/38147133453),
confirmed by the project owner's screenshot; Docker checks took 1 minute 40 seconds.
The first three-job workflow, including the AI/API image, passed on Linux AMD64 in
[run 40157818399](https://github.com/nhphu206/nhabe-radar-himawari-portfolio/actions/runs/40157818399):
Python checks took 1 minute 7 seconds, AI/API checks 1 minute 54 seconds, and preprocessing
Docker checks 1 minute 53 seconds.

1. **Python tests and synthetic sample:** install Python 3.12 and the declared dependencies,
   install the project package, check dependency consistency and syntax, run `doctor`, run
   the test suite, generate a mock sample, and reopen its NPZ/JSON pair.
2. **Docker build and container checks:** after the Python job succeeds, build the image,
   run container tests, persist a mock sample through a writable output mount, reopen it
   through a read-only mount, and report the image size in bytes.
3. **AI tests and forecast API image:** after the Python job succeeds, install CPU-only
   PyTorch and pinned API dependencies, run 30 AI/API tests, create an explicitly synthetic
   one-pair fixture with no future target file, build `Dockerfile.ai`, start the API with
   read-only inputs, exercise the dashboard, pair listing, health/model/predict and visualization
   endpoints over HTTP, verify both PNG dimensions and target-free behavior, reopen the NPZ/JSON
   output, scan the image for excluded data/model files, check dependencies and record image size.

The runner is `ubuntu-24.04` (standard x64 GitHub-hosted runner). The first CI run checked
Linux AMD64 in addition to the Linux ARM64 environment already tested locally. Each job has
a bounded timeout (30 minutes for preprocessing jobs and 35 minutes for AI); new commits
cancel obsolete runs for the same ref.

The workflow needs no user-supplied secrets. GitHub's automatic repository token has only
`contents: read`, and checkout does not persist credentials. No raw observations are downloaded,
no images are published to a registry, and mock outputs are not uploaded as artifacts.

The Linux output-mount smoke test uses the runner's non-root UID/GID so that it can write into
the runner-owned host directory. Other container checks run with the image's default non-root
user. A future real-data run needs separately prepared read-only input mounts.

There is no existing lint configuration, so this initial workflow checks Python syntax with
`compileall` instead of introducing unreviewed style rules. Syntax checking is not linting.

After committing and pushing the workflow, open the repository's **Actions** tab, select
**Radar-Himawari CI**, and inspect all three jobs. A green run confirms those checks for that commit;
it does not confirm real-data gridding, full meteorological quality control, or model performance.

References: [GitHub Actions quickstart](https://docs.github.com/en/actions/get-started/quickstart),
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python).

The extended workflow and synthetic AI fixture were parsed and simulated locally on ARM64:
fixture generation, healthy API container, dashboard/API endpoints, prediction reopen and static
image scan passed. The forecasting extension currently has 67 preprocessing/timing tests and
30 dedicated AI/API tests passing locally. The same contract passed on GitHub's Linux AMD64
runner in run 40157818399. This verifies portability of the synthetic workflow, not model
skill on unseen meteorological observations or real-time production readiness.
