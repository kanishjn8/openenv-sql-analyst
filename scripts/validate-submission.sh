#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run (remote):
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
# Run (local):
#   chmod +x scripts/validate-submission.sh
#   ./scripts/validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
OPENENV_VALIDATE_TIMEOUT=120
OPENENV_VALIDATE_READY_SUBSTR='Ready for multi-mode deployment'
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/health + $PING_URL/reset) ..."

if [ "${SKIP_PING:-0}" = "1" ]; then
  pass "Skipped HF ping (SKIP_PING=1)"
else

# Prefer /health if available (this repo has it)
CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" "$PING_URL/health" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /health"
else
  # Fall back to /reset — NOTE: this repo requires a task_id.
  HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" -d '{"task_id":"sales_summary"}' \
    "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

  if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space is live and responds to /reset"
  elif [ "$HTTP_CODE" = "000" ]; then
    fail "HF Space not reachable (connection failed or timed out)"
    hint "Check your network connection and that the Space is running."
    hint "Try: curl -s -o /dev/null -w '%{http_code}' $PING_URL/health"
    stop_at "Step 1"
  else
    fail "HF Space returned HTTP $HTTP_CODE (expected 200)"
    hint "Make sure your Space is running and the URL is correct."
    hint "Try opening $PING_URL in your browser first."
    stop_at "Step 1"
  fi
fi
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_LOG=$(portable_mktemp "validate-docker")
CLEANUP_FILES+=("$BUILD_LOG")

log "  Building (this can take a few minutes; streaming logs) ..."

# Important: we want the *docker build* exit code, not tee's.
# Using a subshell + pipefail + PIPESTATUS captures docker's exit status reliably.
(
  set -o pipefail
  DOCKER_TIMEOUT_RUNNER=()
  if command -v timeout &>/dev/null; then
    DOCKER_TIMEOUT_RUNNER=(timeout "$DOCKER_BUILD_TIMEOUT")
  elif command -v gtimeout &>/dev/null; then
    DOCKER_TIMEOUT_RUNNER=(gtimeout "$DOCKER_BUILD_TIMEOUT")
  fi

  if [ "${#DOCKER_TIMEOUT_RUNNER[@]}" -gt 0 ]; then
    "${DOCKER_TIMEOUT_RUNNER[@]}" docker build "$DOCKER_CONTEXT" 2>&1 | tee "$BUILD_LOG"
    exit ${PIPESTATUS[0]}
  else
    # Fallback: no timeout/gtimeout available; run directly.
    docker build "$DOCKER_CONTEXT" 2>&1 | tee "$BUILD_LOG"
    exit ${PIPESTATUS[0]}
  fi
)
if [ $? -eq 0 ]; then
  BUILD_OK=true
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  tail -20 "$BUILD_LOG"
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

log "  Running: openenv validate (timeout=${OPENENV_VALIDATE_TIMEOUT}s)"

VALIDATE_LOG=$(portable_mktemp "validate-openenv")
CLEANUP_FILES+=("$VALIDATE_LOG")

# Some openenv versions keep running after printing the readiness line.
# We'll read output line-by-line (no file flush races), mark success once READY is observed,
# then terminate openenv validate so this script can finish.
READY_SEEN=false

FIFO_DIR=$(portable_mktemp "validate-fifo")
rm -f "$FIFO_DIR"
mkdir -p "$FIFO_DIR"
CLEANUP_FILES+=("$FIFO_DIR/out.fifo")
OUT_FIFO="$FIFO_DIR/out.fifo"
mkfifo "$OUT_FIFO"

set +e
(cd "$REPO_DIR" && openenv validate 2>&1 > "$OUT_FIFO") &
OPENENV_PID=$!

START_TS=$(date +%s)
while IFS= read -r line; do
  printf "%s\n" "$line" | tee -a "$VALIDATE_LOG"
  if [[ "$line" == *"$OPENENV_VALIDATE_READY_SUBSTR"* ]]; then
    READY_SEEN=true
    break
  fi

  NOW_TS=$(date +%s)
  if [ $((NOW_TS - START_TS)) -ge "$OPENENV_VALIDATE_TIMEOUT" ]; then
    break
  fi
done < "$OUT_FIFO"

if [ "$READY_SEEN" = true ]; then
  kill "$OPENENV_PID" 2>/dev/null
  wait "$OPENENV_PID" 2>/dev/null
  pass "openenv validate passed"
else
  kill "$OPENENV_PID" 2>/dev/null
  wait "$OPENENV_PID" 2>/dev/null
  fail "openenv validate did not report readiness within ${OPENENV_VALIDATE_TIMEOUT}s"
  tail -200 "$VALIDATE_LOG"
  stop_at "Step 3"
fi
set -e

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
