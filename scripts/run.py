#!/usr/bin/env python3
"""Run the portable pipeline from this checkout, regardless of working directory."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lskp_release.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
