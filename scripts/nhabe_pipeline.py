#!/usr/bin/env python3
"""Repository-local wrapper for the portfolio command-line interface."""

import _bootstrap  # noqa: F401
from nhabe_preprocessing.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
