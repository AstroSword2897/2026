#!/bin/bash
# Test Runner Script for MaxSight 2026 Repository

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

source venv/bin/activate

# Default: run all tests
if [ $# -eq 0 ]; then
    pytest tests/ -v --tb=short
else
    pytest "$@" -v --tb=short
fi
