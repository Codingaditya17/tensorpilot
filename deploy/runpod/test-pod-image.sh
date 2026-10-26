#!/usr/bin/env bash
# Verifies the experiment image without a GPU. Usage: test-pod-image.sh <image> [--slow]
# --slow also waits for the Pod lifetime backstop (about 70 s). Needs docker, ssh and ssh-keygen.
set -uo pipefail
IMAGE="${1:?usage: $0 <image> [--slow]}"
SLOW="${2:-}"
WORK="$(mktemp -d)"
NAME="tp-pod-test-$$"
PORT="${TP_TEST_PORT:-2222}"
FAILS=0
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1" >&2; FAILS=$((FAILS + 1)); }
check() { local label="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$label"; else fail "$label"; fi; }
refuses() { local label="$1"; shift; if "$@" >/dev/null 2>&1; then fail "$label"; else pass "$label"; fi; }

ssh-keygen -q -t ed25519 -N '' -f "$WORK/id" -C pod-test
PUB="$(cat "$WORK/id.pub")"
SSH=(ssh -p "$PORT" -i "$WORK/id" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o LogLevel=ERROR -o ConnectTimeout=5 -o BatchMode=yes)
# A login shell, like an interactive session, so /etc/profile.d is applied.
pod() { "${SSH[@]}" tensorpilot@127.0.0.1 bash -lc "$(printf %q "$*")"; }

refuses "starts only with PUBLIC_KEY (fails closed without it)" docker run --rm "$IMAGE"
refuses "rejects a key carrying options" docker run --rm -e "PUBLIC_KEY=command=\"id\" $PUB" "$IMAGE"
refuses "rejects private key material" docker run --rm -e "PUBLIC_KEY=-----BEGIN OPENSSH PRIVATE KEY-----" "$IMAGE"

mkdir -p "$WORK/vol" && chmod 0755 "$WORK/vol"   # root-owned like a fresh Runpod volume
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:22" -e "PUBLIC_KEY=$PUB" -e TENSORPILOT_POD_MAX_MINUTES=1 \
  -v "$WORK/vol:/workspace" "$IMAGE" >/dev/null || { echo "container did not start" >&2; exit 1; }
for _ in $(seq 1 30); do pod true 2>/dev/null && break; sleep 2; done

check "key login works as tensorpilot" pod true
[[ "$(pod id -u 2>/dev/null)" == 10001 ]] && pass "session runs non-root (uid 10001)" || fail "session runs non-root (uid 10001)"
refuses "root login refused" ssh -p "$PORT" -i "$WORK/id" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=5 root@127.0.0.1 true
refuses "password login refused" ssh -p "$PORT" -o PubkeyAuthentication=no -o PreferredAuthentications=password \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=5 tensorpilot@127.0.0.1 true
check "non-root can write evidence dir" pod 'touch "$TENSORPILOT_RESULTS_ROOT/w" && rm "$TENSORPILOT_RESULTS_ROOT/w"'
check "non-root can write model cache dir" pod 'touch "$TENSORPILOT_CACHE_DIR/w" && rm "$TENSORPILOT_CACHE_DIR/w"'
refuses "scripts and configs are read-only" pod 'touch /opt/tensorpilot/benchmarks/gpu/x'
check "image carries the explicit-device CUDA backend (not the base image copy)" pod 'cd /app && python3 -c "import tensorpilot.cuda_backend as m; assert m.__file__.startswith(\"/usr/local/lib/\"); m.resolve_cuda_device"'
check "benchmark configs shipped" pod 'test -f benchmarks/gpu/v09-first-30min.json && test -f benchmarks/gpu/v09-single-gpu.json'
check "non-secret settings reach SSH sessions" pod '[ -n "$TENSORPILOT_MODEL" ] && [ "$TENSORPILOT_REVISION" = 7ae557604adf67be50417f59c2c2f167def9a775 ] && [ "$HF_HOME" = /workspace/huggingface ] && [ "$TENSORPILOT_CACHE_DIR" = /workspace/huggingface ] && [ "$TENSORPILOT_RESULTS_ROOT" = /workspace/results ]'
refuses "API keys are not forwarded or stored" pod 'env | grep -q API_KEY || grep -rq API_KEY /etc/profile.d'
check "first-run.sh --plan validates" pod 'first-run.sh --plan'
LISTEN="$(pod "python3 -c \"
import re
ports=sorted({int(l.split()[1].split(':')[1],16) for f in ('/proc/net/tcp','/proc/net/tcp6') for l in open(f).read().splitlines()[1:] if l.split()[3]=='0A'})
print(ports)\"" 2>/dev/null)"
[[ "$LISTEN" == "[22]" ]] && pass "only port 22 is listening ($LISTEN)" || fail "only port 22 is listening (got $LISTEN)"

# No GPU here, so the session must stop at preflight and still leave a verified archive.
pod 'TENSORPILOT_SESSION_CAP=5m first-run.sh' >"$WORK/run.log" 2>&1
RC=$?
[[ $RC -ne 0 ]] && pass "session stops without a GPU (exit $RC)" || fail "session stops without a GPU"
check "partial evidence archived with checksum after failure" pod 'ls "$TENSORPILOT_RESULTS_ROOT"/*.tar.gz "$TENSORPILOT_RESULTS_ROOT"/*.tar.gz.sha256'
check "archive passes tar integrity check" pod 'tar -tzf "$TENSORPILOT_RESULTS_ROOT"/*.tar.gz'
check "archive holds diagnostics and the preflight failure" pod 'tar -tzf "$TENSORPILOT_RESULTS_ROOT"/*.tar.gz | grep -q diagnostics/system.txt && tar -tzf "$TENSORPILOT_RESULTS_ROOT"/*.tar.gz | grep -q preflight.json'
pod 'TENSORPILOT_SESSION_CAP=2s first-run.sh' >"$WORK/cap.log" 2>&1
[[ $? -ne 0 ]] && pass "session cap fires and the run ends" || fail "session cap fires and the run ends"

START=$SECONDS
docker stop -t 20 "$NAME" >/dev/null 2>&1
(( SECONDS - START < 15 )) && pass "graceful stop under 15 s" || fail "graceful stop under 15 s"
[[ "$(docker inspect -f '{{.State.ExitCode}}' "$NAME")" == 0 ]] && pass "clean exit code 0 on SIGTERM" || fail "clean exit code 0 on SIGTERM"

if [[ "$SLOW" == "--slow" ]]; then
  docker run -d --name "$NAME-life" -e "PUBLIC_KEY=$PUB" -e TENSORPILOT_POD_MAX_MINUTES=1 "$IMAGE" >/dev/null
  sleep 75
  [[ "$(docker inspect -f '{{.State.Status}}' "$NAME-life")" == exited ]] && pass "lifetime backstop exits the container" || fail "lifetime backstop exits the container"
  docker rm -f "$NAME-life" >/dev/null 2>&1
fi

# Regression: a mounted volume that refuses chown (the Runpod failure: "chown: ... Operation not permitted").
# Dropping CAP_CHOWN makes root's chown fail with the same EPERM. Named volumes, like Runpod's, start root-owned.
wait_ssh() { for _ in $(seq 1 30); do pod true 2>/dev/null && return 0; sleep 2; done; return 1; }
VOLS=()
mkvol() { local v="tp-pod-test-$1-$$"; docker volume create "$v" >/dev/null; VOLS+=("$v"); echo "$v"; }
cleanup_volumes() { for v in ${VOLS[@]+"${VOLS[@]}"}; do docker volume rm -f "$v" >/dev/null 2>&1 || true; done; }
trap 'cleanup; cleanup_volumes' EXIT

echo "--- scenario: chown refused, volume root-owned (falls back to container disk)"
V="$(mkvol a)"
docker run -d --name "$NAME-a" --cap-drop CHOWN -p "127.0.0.1:$PORT:22" -e "PUBLIC_KEY=$PUB" -v "$V:/workspace" "$IMAGE" >/dev/null
wait_ssh && pass "container starts although chown is refused" || fail "container starts although chown is refused"
LOGS="$(docker logs "$NAME-a" 2>&1)"
grep -q "chown of /workspace/results refused" <<<"$LOGS" && pass "refusal is reported, not hidden" || fail "refusal is reported, not hidden"
grep -q "storage diagnostic for /workspace/results" <<<"$LOGS" && pass "diagnostic printed for the unusable volume" || fail "diagnostic printed for the unusable volume"
grep -q "WARNING results -> /home/tensorpilot/work/results" <<<"$LOGS" && pass "fallback announced as a warning" || fail "fallback announced as a warning"
[[ "$(pod 'echo "$TENSORPILOT_RESULTS_ROOT $TENSORPILOT_CACHE_DIR $HF_HOME"' 2>/dev/null)" == "/home/tensorpilot/work/results /home/tensorpilot/work/huggingface /home/tensorpilot/work/huggingface" ]] \
  && pass "results, cache and HF_HOME all follow the fallback" || fail "results, cache and HF_HOME all follow the fallback"
check "non-root really writes to the fallback evidence dir" pod 'touch "$TENSORPILOT_RESULTS_ROOT/w" && rm "$TENSORPILOT_RESULTS_ROOT/w"'
pod 'first-run.sh --plan' 2>/dev/null | grep -q "/home/tensorpilot/work/results/v09-" && pass "first-run.sh plan uses the fallback results path" || fail "first-run.sh plan uses the fallback results path"
pod 'TENSORPILOT_SESSION_CAP=5m first-run.sh' >/dev/null 2>&1
check "evidence archived under the fallback path after a failed run" pod 'ls "$TENSORPILOT_RESULTS_ROOT"/*.tar.gz.sha256'
docker rm -f "$NAME-a" >/dev/null 2>&1

echo "--- scenario: chown refused but the volume is already owned by the user (uses the volume)"
V="$(mkvol b)"
docker run --rm -v "$V:/w" --entrypoint chown "$IMAGE" 10001:10001 /w
docker run -d --name "$NAME-b" --cap-drop CHOWN -p "127.0.0.1:$PORT:22" -e "PUBLIC_KEY=$PUB" -v "$V:/workspace" "$IMAGE" >/dev/null
wait_ssh && pass "container starts with a pre-owned volume" || fail "container starts with a pre-owned volume"
[[ "$(pod 'echo "$TENSORPILOT_RESULTS_ROOT $TENSORPILOT_CACHE_DIR"' 2>/dev/null)" == "/workspace/results /workspace/huggingface" ]] \
  && pass "mounted volume used when it is verifiably writable" || fail "mounted volume used when it is verifiably writable"
check "non-root writes to the mounted volume" pod 'touch "$TENSORPILOT_RESULTS_ROOT/w" && rm "$TENSORPILOT_RESULTS_ROOT/w"'
docker rm -f "$NAME-b" >/dev/null 2>&1

echo "--- scenario: nothing writable anywhere (must fail with a clear diagnostic)"
V="$(mkvol c)"
OUT="$(docker run --rm --read-only --cap-drop CHOWN -e "PUBLIC_KEY=$PUB" -v "$V:/workspace" "$IMAGE" 2>&1)"; RC=$?
[[ $RC -ne 0 ]] && pass "exits non-zero when no storage is writable (exit $RC)" || fail "exits non-zero when no storage is writable"
grep -q "no writable storage for results" <<<"$OUT" && pass "failure names the problem and both locations" || fail "failure names the problem and both locations"
echo "failures: $FAILS"
exit $(( FAILS > 0 ))
