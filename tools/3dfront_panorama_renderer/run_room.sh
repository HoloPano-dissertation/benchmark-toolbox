#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 ROOM_DIR OUTPUT_DIR [renderer options]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOM_DIR="$1"
OUTPUT_DIR="$2"
shift 2

BLENDERPROC_BIN="${BLENDERPROC_BIN:-blenderproc}"
cleanup_temp=true
if [[ -n "${BLENDERPROC_TEMP_DIR:-}" ]]; then
    RUN_TEMP="$BLENDERPROC_TEMP_DIR"
    mkdir -p "$RUN_TEMP"
    cleanup_temp=false
else
    RUN_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/front3d-panorama.XXXXXX")"
fi
cleanup() {
    if $cleanup_temp && [[ -d "$RUN_TEMP" ]]; then
        rm -rf -- "$RUN_TEMP"
    fi
}
trap cleanup EXIT

command=("$BLENDERPROC_BIN" run "$SCRIPT_DIR/render.py")
if [[ -n "${BLENDER_INSTALL_PATH:-}" ]]; then
    command+=(--blender-install-path "$BLENDER_INSTALL_PATH")
fi
command+=(--temp-dir "$RUN_TEMP" "$ROOM_DIR" "$OUTPUT_DIR" "$@")
"${command[@]}"
