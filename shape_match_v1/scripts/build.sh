#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build_dir="${BUILD_DIR:-build}"
mkdir -p "$build_dir"
compiler="${CXX:-g++}"
flags=(-std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Iinclude -Isrc -pthread)
if [[ "${SANITIZE:-0}" == 1 ]]; then
  flags=(-std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude -Isrc -pthread)
fi
sources=(src/types.cpp src/model.cpp src/geometry.cpp src/search.cpp src/result.cpp)
objects=()
for source in "${sources[@]}"; do
  object="$build_dir/$(basename "${source%.cpp}").o"
  "$compiler" "${flags[@]}" -c "$source" -o "$object"
  objects+=("$object")
done
ar rcs "$build_dir/libshape_match.a" "${objects[@]}"
for entry in examples/match_sample.cpp tests/test_match.cpp tests/accuracy_bench.cpp tests/performance_bench.cpp; do
  "$compiler" "${flags[@]}" "$entry" "$build_dir/libshape_match.a" -o "$build_dir/$(basename "${entry%.cpp}")"
done
echo "Built library, sample, tests and accuracy benchmark in $build_dir/"
