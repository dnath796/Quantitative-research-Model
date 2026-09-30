# dpe — Derivatives Pricing Engine (C++17 port).
# Targets: dpe (static library), demo, dpe_tests (GoogleTest, run via ctest).

cmake_minimum_required(VERSION 3.16)
project(dpe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(dpe
  src/black_scholes.cpp
  src/binomial.cpp
  src/monte_carlo.cpp
  src/fx.cpp
  src/utils.cpp
  src/validation.cpp
)
target_include_directories(dpe PUBLIC include)

# ---------------------------------------------------------------------------
# Demo (reads the shared CSVs from ../data)
# ---------------------------------------------------------------------------
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE dpe)
target_compile_definitions(demo PRIVATE
  DPE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")

# ---------------------------------------------------------------------------
# Tests (golden suite loads ../data/golden/golden.json)
# ---------------------------------------------------------------------------
enable_testing()
find_package(GTest REQUIRED)

add_executable(dpe_tests
  tests/test_black_scholes.cpp
  tests/test_implied_vol.cpp
  tests/test_binomial.cpp
  tests/test_monte_carlo.cpp
  tests/test_fx_utils.cpp
  tests/test_golden.cpp
)
target_link_libraries(dpe_tests PRIVATE dpe GTest::gtest GTest::gtest_main)
target_compile_definitions(dpe_tests PRIVATE
  DPE_GOLDEN_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../data/golden/golden.json")

include(GoogleTest)
gtest_discover_tests(dpe_tests)
