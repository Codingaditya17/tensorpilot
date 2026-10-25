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
check "non-root can write evidence dir" pod 'touch /workspace/results/w && rm /workspace/results/w'
check "non-root can write model cache dir" pod 'touch /workspace/huggingface/w && rm /workspace/huggingface/w'
refuses "scripts and configs are read-only" pod 'touch /opt/tensorpilot/benchmarks/gpu/x'
check "benchmark configs shipped" pod 'test -f benchmarks/gpu/v09-first-30min.json && test -f benchmarks/gpu/v09-single-gpu.json'
check "non-secret settings reach SSH sessions" pod '[ -n "$TENSORPILOT_MODEL" ] && [ "$TENSORPILOT_REVISION" = 7ae557604adf67be50417f59c2c2f167def9a775 ] && [ "$HF_HOME" = /workspace/huggingface ]'
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
check "partial evidence archived with checksum after failure" pod 'ls /workspace/results/*.tar.gz /workspace/results/*.tar.gz.sha256'
check "archive passes tar integrity check" pod 'tar -tzf /workspace/results/*.tar.gz'
check "archive holds diagnostics and the preflight failure" pod 'tar -tzf /workspace/results/*.tar.gz | grep -q diagnostics/system.txt && tar -tzf /workspace/results/*.tar.gz | grep -q preflight.json'
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
echo "failures: $FAILS"
exit $(( FAILS > 0 ))
