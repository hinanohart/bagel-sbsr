#!/usr/bin/env bash
# Clone the BAGEL upstream Python source into vendor/bagel-upstream/.
# We do not git-submodule it (see docs/ARCH.md for reasoning).
# Re-running is idempotent: existing clone is `git pull` updated unless --pin REV is given.

set -euo pipefail

REPO_URL="${BAGEL_REPO_URL:-https://github.com/ByteDance-Seed/Bagel.git}"
DEST="${BAGEL_SRC_DEST:-vendor/bagel-upstream}"
PIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pin)
      PIN="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$DEST")"

if [[ -d "$DEST/.git" ]]; then
  echo "Updating existing clone at $DEST"
  git -C "$DEST" fetch --depth 1 origin
  if [[ -n "$PIN" ]]; then
    git -C "$DEST" checkout --quiet "$PIN"
  else
    git -C "$DEST" reset --hard origin/HEAD
  fi
else
  echo "Cloning $REPO_URL -> $DEST"
  if [[ -n "$PIN" ]]; then
    git clone --depth 1 "$REPO_URL" "$DEST"
    git -C "$DEST" fetch --depth 1 origin "$PIN"
    git -C "$DEST" checkout --quiet "$PIN"
  else
    git clone --depth 1 "$REPO_URL" "$DEST"
  fi
fi

echo "BAGEL src installed at $DEST (HEAD: $(git -C "$DEST" rev-parse --short HEAD))"
echo "NOTE: $DEST is Apache-2.0; preserve its LICENSE file when redistributing."
echo "Add it to PYTHONPATH before importing BAGEL classes:"
echo "  export PYTHONPATH=\"$(pwd)/$DEST:\${PYTHONPATH:-}\""
