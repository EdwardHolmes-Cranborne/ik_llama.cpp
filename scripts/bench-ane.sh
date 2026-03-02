#!/bin/bash
# bench-ane.sh — ANE Graph-Split Accelerator Benchmark Suite
# Runs hardware characterization + performance comparison across:
#   GPU-only, ANE-only, GPU+ANE concurrent, GPU+ANE+CPU
#
# Usage:
#   ./scripts/bench-ane.sh [model.gguf] [--quick] [--build-only] [--skip-build]
#
# The script will:
#   1. Build ik_llama.cpp with GGML_ANE=ON + GGML_METAL=ON
#   2. Run Phase 0 hardware characterization (test-ane-hardware)
#   3. Run Phase 1 backend tests (test-ane-backend)
#   4. Run performance benchmarks comparing GPU/ANE/GPU+ANE modes
#
# Results are saved to docs/m3_ane_characterization.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build-ane"
RESULTS_DIR="${PROJECT_DIR}/docs"
RESULTS_FILE="${RESULTS_DIR}/m3_ane_characterization.md"

# Parse arguments
MODEL_PATH=""
QUICK_MODE=""
BUILD_ONLY=false
SKIP_BUILD=false

for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_MODE="--quick" ;;
        --build-only) BUILD_ONLY=true ;;
        --skip-build) SKIP_BUILD=true ;;
        *.gguf) MODEL_PATH="$arg" ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[ANE-BENCH]${NC} $*"; }
ok()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ===========================================================================
# Step 1: Build
# ===========================================================================
if [ "$SKIP_BUILD" = false ]; then
    log "Building ik_llama.cpp with GGML_ANE=ON + GGML_METAL=ON..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake "$PROJECT_DIR" \
        -DGGML_ANE=ON \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=ON \
        2>&1 | tail -20

    cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -30

    if [ $? -ne 0 ]; then
        fail "Build failed!"
        exit 1
    fi
    ok "Build succeeded"
    cd "$PROJECT_DIR"
fi

if [ "$BUILD_ONLY" = true ]; then
    log "Build-only mode, exiting."
    exit 0
fi

# ===========================================================================
# Step 2: Hardware Characterization (Phase 0)
# ===========================================================================
log "============================================"
log "Phase 0: M3 ANE Hardware Characterization"
log "============================================"

mkdir -p "$RESULTS_DIR"
{
    echo "# M3 ANE Characterization Results"
    echo ""
    echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    echo "macOS: $(sw_vers -productVersion)"
    echo ""
} > "$RESULTS_FILE"

HW_TEST="${BUILD_DIR}/bin/test-ane-hardware"
if [ -f "$HW_TEST" ]; then
    log "Running hardware tests..."
    {
        echo "## Phase 0: Hardware Tests"
        echo ""
        echo '```'
    } >> "$RESULTS_FILE"

    "$HW_TEST" $QUICK_MODE 2>&1 | tee -a "$RESULTS_FILE"
    HW_EXIT=$?
    echo '```' >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    if [ $HW_EXIT -eq 0 ]; then
        ok "Hardware tests passed"
    else
        warn "Some hardware tests failed (exit code: $HW_EXIT)"
        warn "Check $RESULTS_FILE for details"
    fi
else
    warn "test-ane-hardware not found at $HW_TEST"
fi

# ===========================================================================
# Step 3: Backend Tests (Phase 1)
# ===========================================================================
log "============================================"
log "Phase 1: ANE Backend Tests"
log "============================================"

BE_TEST="${BUILD_DIR}/bin/test-ane-backend"
if [ -f "$BE_TEST" ]; then
    log "Running backend tests..."
    {
        echo "## Phase 1: Backend Tests"
        echo ""
        echo '```'
    } >> "$RESULTS_FILE"

    "$BE_TEST" 2>&1 | tee -a "$RESULTS_FILE"
    BE_EXIT=$?
    echo '```' >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    if [ $BE_EXIT -eq 0 ]; then
        ok "Backend tests passed"
    else
        warn "Some backend tests failed (exit code: $BE_EXIT)"
    fi
else
    warn "test-ane-backend not found at $BE_TEST"
fi

# ===========================================================================
# Step 4: Model Benchmarks (if model provided)
# ===========================================================================
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    log "============================================"
    log "Performance Benchmarks: $MODEL_PATH"
    log "============================================"

    LLAMA_BENCH="${BUILD_DIR}/bin/llama-bench"
    LLAMA_CLI="${BUILD_DIR}/bin/llama-cli"

    {
        echo "## Performance Benchmarks"
        echo ""
        echo "Model: \`$(basename "$MODEL_PATH")\`"
        echo ""
    } >> "$RESULTS_FILE"

    if [ -f "$LLAMA_BENCH" ]; then
        log "Running llama-bench (GPU-only baseline)..."
        {
            echo "### GPU-only (Metal) Baseline"
            echo ""
            echo '```'
        } >> "$RESULTS_FILE"

        "$LLAMA_BENCH" -m "$MODEL_PATH" -p 512,1024,2048 -n 0 -ngl 999 2>&1 | tee -a "$RESULTS_FILE"
        echo '```' >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    else
        warn "llama-bench not found"
    fi

    if [ -f "$LLAMA_CLI" ]; then
        log "Running short inference test (GPU-only)..."
        PROMPT="The quick brown fox jumps over the lazy dog. In the world of artificial intelligence and machine learning, transformer architectures have revolutionized natural language processing."

        {
            echo "### Inference Test"
            echo ""
            echo '```'
        } >> "$RESULTS_FILE"

        "$LLAMA_CLI" -m "$MODEL_PATH" -p "$PROMPT" -n 32 --no-display-prompt -ngl 999 2>&1 | tee -a "$RESULTS_FILE"
        echo '```' >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    fi
else
    if [ -n "$MODEL_PATH" ]; then
        warn "Model not found: $MODEL_PATH"
    else
        log "No model provided. Skipping performance benchmarks."
        log "Usage: ./scripts/bench-ane.sh /path/to/model.gguf"
    fi
fi

# ===========================================================================
# Summary
# ===========================================================================
log "============================================"
log "Results saved to: $RESULTS_FILE"
log "============================================"

{
    echo "## Stage Gate Assessment"
    echo ""
    echo "| Criterion | Result | Action |"
    echo "|-----------|--------|--------|"
    echo "| Private APIs work | See Phase 0 above | |"
    echo "| Metal+ANE concurrent | See concurrency test | |"
    echo "| Peak TFLOPS > 3 | See TFLOPS sweep | |"
    echo "| Backend tests pass | See Phase 1 above | |"
    echo ""
} >> "$RESULTS_FILE"

log "Done!"
