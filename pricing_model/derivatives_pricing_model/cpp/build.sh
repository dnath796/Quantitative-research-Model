#!/usr/bin/env bash
# Build the dpe C++ port (library + demo + tests).
# Tests:  ctest --test-dir build --output-on-failure
# Demo:   ./build/demo
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
