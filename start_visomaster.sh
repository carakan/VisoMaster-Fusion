#!/bin/bash
# VisoMaster Fusion launcher — runs the app in THIS directory using the local
# .venv (Python 3.13, torch 2.14.1 custom MPS build) on the AMD RX 6800 XT.
# This is the operational script.
#
# Usage (run it from anywhere; it always launches this directory's app):
#   ./start_visomaster.sh            start the GUI (default)
#   ./start_visomaster.sh start      same as above
#   ./start_visomaster.sh stop       stop the app + any strays
#   ./start_visomaster.sh status     show running/state
#   ./start_visomaster.sh log        tail the log
#
# ---------------------------------------------------------------------------
# CHANGELOG. Newest last. Read this before changing anything here; every line
# records WHY it exists and what breaks without it.
#
# 2026-10-15  (initial)
#   Runs `.venv/bin/python main.py` from the repo root. Running from the repo
#   root is REQUIRED: main.py loads its Qt stylesheet relative to the working
#   directory, so launching from elsewhere renders an unstyled window.
#
#   No provider flag is passed. Provider is chosen in the UI under
#   "Providers Priority" and MUST be CPU on this machine. Measured, per model:
#     ArcFace embed   CPU 45 ms  vs CoreML 601 ms
#     face parse      CPU 79 ms  vs CoreML 839 ms
#     face swap       CPU 178 ms vs CoreML 334 ms
#     inswapper fp16  CPU 433 ms vs CoreML 534 ms
#   CoreML must run with RequireStaticInputShapes=1, and these graphs are
#   dynamic, so CoreML falls back to the CPU EP and just adds partitioning
#   overhead. Do not "upgrade" the provider to CoreML here.
#
#   ONNX Runtime has no MPS execution provider at all, and inference binds
#   torch tensors straight into ORT via io_binding(buffer_ptr=...), so an MPS
#   tensor would be bound as garbage. On its own that means the GPU cannot
#   accelerate these models.
#
# 2026-10-16  (MPS swapper backend)
#   The GPU now DOES accelerate the face swappers, via
#   app/processors/utils/mps_backend.py, which converts the ONNX graph with
#   onnx2torch so torch owns the buffers and the graph runs natively on MPS.
#   Controlled by Settings > General > "Providers Priority": pick "MPS" to run
#   them on the GPU, anything else to leave them on ONNX Runtime.
#   VISOMASTER_MPS_BACKEND=off|auto presets that choice for a run started
#   outside the GUI.
#
#   Only the models in mps_backend.MPS_MODELS move: the swappers and the SCRFD
#   detectors. The face embedders stay on ONNX Runtime because CoreML already
#   beats MPS on them (w600k_r50: CoreML 6 ms vs MPS 23 ms) -- MPS dispatch
#   overhead exceeds the arithmetic at 112x112. Measured on the swappers:
#     Inswapper128              CPU 560 ms -> MPS  52 ms
#     InStyleSwapper256 Version A CPU 2426 ms -> MPS 143 ms
#   Provider priority still must be CPU: that governs the models which stay on
#   ONNX Runtime, and CoreML loses to CPU on this machine for all of them.
# ---------------------------------------------------------------------------

set -u

VM_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$VM_DIR/.venv/bin/python"
LOG_FILE="$VM_DIR/visomaster.log"
PID_FILE="$VM_DIR/visomaster.pid"
READY_TIMEOUT=300

running_pids() {
    pgrep -f "$VM_DIR/.venv/bin/python main.py" 2>/dev/null
}

pidfile_pid() {
    [ -f "$PID_FILE" ] || return 0
    local p
    p=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        echo "$p"
    else
        rm -f "$PID_FILE"
    fi
}

# The GUI window is the readiness signal. PySide6 prints its font-alias
# warning during Qt init, so that line means the toolkit came up.
ready() {
    grep -q "qt.qpa.fonts" "$LOG_FILE" 2>/dev/null
}

launch() {
    local existing
    existing=$(running_pids)
    if [ -n "$existing" ]; then
        echo "$existing" | head -1 >"$PID_FILE"
        echo "already running (pid $(echo "$existing" | head -1))"
        return 0
    fi

    if [ ! -x "$VENV_PY" ]; then
        echo "error: venv missing at $VENV_PY" >&2
        echo "       see LOCAL-NOTES.md for how to rebuild it" >&2
        return 1
    fi

    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "warning: ffmpeg not on PATH - recording and video export will fail."
        echo "         install with: brew install ffmpeg"
    fi

    cd "$VM_DIR" || return 1
    nohup "$VENV_PY" main.py "$@" >"$LOG_FILE" 2>&1 &
    local child=$!
    echo "$child" >"$PID_FILE"
    echo "starting (pid $child), log: $LOG_FILE"

    local i=0
    while [ "$i" -lt "$READY_TIMEOUT" ]; do
        if ! kill -0 "$child" 2>/dev/null; then
            echo "failed to start - see $LOG_FILE and crash_logs/"
            return 1
        fi
        if ready; then
            echo "ready: VisoMaster Fusion GUI (pid $child)"
            return 0
        fi
        sleep 3
        i=$((i + 3))
    done
    echo "window not detected after ${READY_TIMEOUT}s - still running (pid $child)"
    return 0
}

case "${1:-}" in
    stop)
        pids=$(running_pids)
        [ -z "$pids" ] && pids=$(pidfile_pid)
        [ -z "$pids" ] && { echo "not running"; rm -f "$PID_FILE"; exit 0; }
        for p in $pids; do kill "$p" 2>/dev/null; done
        sleep 3
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
        sleep 1
        rm -f "$PID_FILE"
        left=$(running_pids)
        if [ -z "$left" ]; then echo "stopped"; else echo "warning: still alive: $left"; fi
        ;;
    status)
        pids=$(running_pids)
        if [ -n "$pids" ]; then
            echo "running (pid $(echo "$pids" | head -1))"
        else
            echo "not running"
        fi
        ;;
    log)
        tail -n "${2:-60}" "$LOG_FILE"
        ;;
    start)
        shift
        launch "$@"
        ;;
    "")
        launch
        ;;
    *)
        echo "usage: $0 [start|stop|status|log]" >&2
        exit 2
        ;;
esac