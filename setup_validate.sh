#!/usr/bin/env bash
#
# setup_validate.sh — Submission validation launcher.
#
# Accepts either:
#   1) HF runtime URL: https://<owner>-<space>.hf.space
#   2) HF Space page URL: https://huggingface.co/spaces/<owner>/<space>
#
# Usage:
#   chmod +x setup_validate.sh
#   ./setup_validate.sh <space_url_or_ping_url> [repo_dir]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${SCRIPT_DIR}/scripts/validate-submission.sh"

if [ "${1:-}" = "" ]; then
  echo "Usage: $0 <space_url_or_ping_url> [repo_dir]"
  echo ""
  echo "Examples:"
  echo "  $0 https://kanishjn8-openenv-sql-analyst.hf.space"
  echo "  $0 https://huggingface.co/spaces/kanishjn8/openenv-sql-analyst"
  exit 1
fi

RAW_URL="${1%/}"
REPO_DIR="${2:-.}"
PING_URL="$RAW_URL"

# Convert Space page URL to runtime URL expected by validator.
if [[ "$RAW_URL" =~ ^https://huggingface\.co/spaces/([^/]+)/([^/]+)$ ]]; then
  OWNER="${BASH_REMATCH[1]}"
  SPACE="${BASH_REMATCH[2]}"
  PING_URL="https://${OWNER}-${SPACE}.hf.space"
fi

echo "[setup] using ping url: ${PING_URL}"
echo "[setup] note: '/?logs=container 404' in HF logs is expected and non-fatal."

exec "$VALIDATOR" "$PING_URL" "$REPO_DIR"
