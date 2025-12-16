#!/usr/bin/env bash
echo "[deprecated] Use ./run_stack.sh instead." >&2
exec "$(dirname "$0")/../run_stack.sh" "$@"
