# Continuous integration

The workflow in `.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch.
Its first hosted run is pending; local ARM64 Docker verification is documented separately.

1. **Python tests and synthetic sample:** install Python 3.12 and the declared dependencies,
   install the project package, check dependency consistency and syntax, run `doctor`, run
   the test suite, generate a mock sample, and reopen its NPZ/JSON pair.
2. **Docker build and container checks:** after the Python job succeeds, build the image,
   run container tests, persist a mock sample through a writable output mount, reopen it
   through a read-only mount, and report the image size in bytes.

The runner is `ubuntu-24.04` (standard x64 GitHub-hosted runner). The first CI run will check
Linux AMD64 in addition to the Linux ARM64 environment already tested locally. Each job has
a 30-minute timeout; new commits cancel obsolete runs for the same ref.

The workflow needs no user-supplied secrets. GitHub's automatic repository token has only
`contents: read`, and checkout does not persist credentials. No raw observations are downloaded,
no images are published to a registry, and mock outputs are not uploaded as artifacts.

The Linux output-mount smoke test uses the runner's non-root UID/GID so that it can write into
the runner-owned host directory. Other container checks run with the image's default non-root
user. A future real-data run needs separately prepared read-only input mounts.

There is no existing lint configuration, so this initial workflow checks Python syntax with
`compileall` instead of introducing unreviewed style rules. Syntax checking is not linting.

After committing and pushing the workflow, open the repository's **Actions** tab, select
**Preprocessing CI**, and inspect both jobs. A green run confirms those checks for that commit;
it does not confirm real-data gridding, full meteorological quality control, or model performance.

References: [GitHub Actions quickstart](https://docs.github.com/en/actions/get-started/quickstart),
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python).
