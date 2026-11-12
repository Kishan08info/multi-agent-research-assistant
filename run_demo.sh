#!/usr/bin/env bash
# Run the demo question and the full eval set.
set -e
cd "$(dirname "$0")"
python3 demo.py
echo "---"
python3 -m research.cli eval
