#!/usr/bin/env bash
set -euo pipefail

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    for key_type in rsa ecdsa ed25519; do
        key_file="/etc/ssh/ssh_host_${key_type}_key"
        if [ ! -f "$key_file" ]; then
            ssh-keygen -t "$key_type" -f "$key_file" -q -N ''
        fi
    done

    service ssh start && echo "load-balancer-worker: SSH server started" || echo "load-balancer-worker: SSH server could not be started" >&2
fi

TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
if [ -n "${TCMALLOC}" ]; then
    export LD_PRELOAD="${TCMALLOC}"
fi

echo "load-balancer-worker: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    print(f'OK: {name}')
except Exception as e:
    print(f'FAIL: {e}')
    raise
" 2>&1); then
    echo "load-balancer-worker: GPU is not available. PyTorch CUDA init failed:"
    echo "load-balancer-worker: $GPU_CHECK"
    exit 1
fi
echo "load-balancer-worker: GPU available - $GPU_CHECK"

comfy-manager-set-mode offline || echo "load-balancer-worker: Could not set ComfyUI-Manager network_mode" >&2

: "${COMFY_LOG_LEVEL:=INFO}"
: "${COMFY_PID_FILE:=/tmp/comfyui.pid}"
: "${COMFY_BOOT_GRACE_PERIOD_S:=5}"
: "${PORT:=80}"
: "${PORT_HEALTH:=${PORT}}"
: "${GRANIAN_HTTP:=auto}"
: "${GRANIAN_WORKERS:=1}"
: "${GRANIAN_LOG_LEVEL:=info}"

echo "load-balancer-worker: Starting ComfyUI"
python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
echo $! > "${COMFY_PID_FILE}"

sleep "${COMFY_BOOT_GRACE_PERIOD_S}"
if ! kill -0 "$(cat "${COMFY_PID_FILE}")" 2>/dev/null; then
    echo "load-balancer-worker: ComfyUI exited during startup. Failing fast so the platform does not keep routing to an initializing worker." >&2
    exit 1
fi

granian_args=(
    --interface asgi
    --host 0.0.0.0
    --http "${GRANIAN_HTTP}"
    --workers "${GRANIAN_WORKERS}"
    --log-level "${GRANIAN_LOG_LEVEL}"
)

if [ "${PORT_HEALTH}" = "${PORT}" ]; then
    echo "load-balancer-worker: Starting Granian on port ${PORT}"
    exec granian "${granian_args[@]}" --port "${PORT}" src.server:main_app
fi

cleanup() {
    if [ -n "${main_pid:-}" ] && kill -0 "${main_pid}" 2>/dev/null; then
        kill "${main_pid}" 2>/dev/null || true
        wait "${main_pid}" 2>/dev/null || true
    fi

    if [ -n "${health_pid:-}" ] && kill -0 "${health_pid}" 2>/dev/null; then
        kill "${health_pid}" 2>/dev/null || true
        wait "${health_pid}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "load-balancer-worker: Starting Granian health server on port ${PORT_HEALTH}"
granian "${granian_args[@]}" --port "${PORT_HEALTH}" src.server:health_app &
health_pid=$!

echo "load-balancer-worker: Starting Granian main server on port ${PORT}"
granian "${granian_args[@]}" --port "${PORT}" src.server:main_app &
main_pid=$!

wait -n "${main_pid}" "${health_pid}"
status=$?
exit "${status}"
