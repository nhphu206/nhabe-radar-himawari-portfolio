# Historical validation evidence

These small files are copies of verified local outputs from the August 2025 research pipeline:

| File | Purpose |
| --- | --- |
| `dataset_audit.json` | Full-dataset audit: 8,774 samples, 4,387 groups, PASS |
| `daily_summary.csv` | Daily counts and descriptive quality summaries |
| `dataset_card.json` | Dataset shape, band order, configuration and split policy |

The originals remain in `output/audit/` and `output/dataset/`. The relative output paths and
index checksum in the dataset card describe the local research run; those referenced files
are not bundled here. These snapshots do not imply that the portfolio Docker workflow has
already been verified, or that meteorological quality control or AI training is complete.

The full `index.csv`, `time_manifest.csv`, and machine-specific `provenance.json` stay local.
No observation arrays or raw data are stored in this directory.
