#!/usr/bin/env bash
# Print export lines for three distinct random API keys. Use: eval "$(pod-keys.sh)"
# Keys exist only in that shell's environment: never in Runpod settings, files or the evidence.
set -euo pipefail
python3 - <<'PY'
import secrets
for name in ("TENSORPILOT_API_KEY", "TENSORPILOT_CONTROL_API_KEY", "TENSORPILOT_DATA_API_KEY"):
    print(f"export {name}={secrets.token_urlsafe(32)}")
PY
