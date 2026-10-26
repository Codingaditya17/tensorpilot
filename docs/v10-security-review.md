# v1.0 release candidate: security review

What was checked before the experiment image was published, what it found, and what is accepted. Not a penetration test; nothing here was run on a GPU or against a rented Pod.

## Checks run

| Check | Result |
|---|---|
| Full test suite locally, on the image's own package in the Linux container, and the Pod-image suite (SSH hardening, non-root, only port 22, key-only login, read-only scripts) | pass (see the release notes for counts) |
| `ruff`, ShellCheck (`--severity=warning`) on every Pod script, hadolint on both Dockerfiles | clean |
| Secret scan of the working tree, the new history since `583f8b0`, untracked files and the preserved baseline archive (GitHub, Hugging Face, OpenAI-style, AWS, Runpod, Slack, private-key and long `Bearer` patterns) | no secrets. The only matches are deliberately fake fixtures in tests and a test that feeds a private-key header to check it is refused |
| Archives refuse to be written if they contain the API key; keys travel by environment only | tested, including a refusal test |
| `bandit` on `tensorpilot/` (medium and above) | 9 findings, all reviewed: 6 × B615 `from_pretrained` without a revision (false positives: every call loads a local snapshot already resolved at a pinned revision, or reads `config.json` with a pinned revision); 3 × B310 `urlopen` on a URL (the two this milestone touched now reject non-http(s) schemes; the third is the container health check with a hard-coded `http://127.0.0.1` URL) |
| `pip-audit` on the three requirements files | see below |
| Dangerous constructs in the package (`torch.load`, `pickle`, `torch.jit`, `torch.export`, `trust_remote_code`, `shell=True`, `os.system`, `eval`) | none; subprocesses use argument lists |

## Advisories found by `pip-audit`, and why they are accepted for this candidate

| Package | Advisory | Affects | Why accepted | Mitigation |
|---|---|---|---|---|
| torch 2.9.1 | CVE-2026-24747 (fixed in 2.10.0): memory corruption in the `weights_only` unpickler when loading a malicious `.pth` | loading untrusted pickle checkpoints | TensorPilot never calls `torch.load`; weights come from `.safetensors` files at a pinned Hugging Face revision | `trust_remote_code` is never set; the model cache is the Pod's own volume |
| torch 2.9.1 | CVE-2025-3000 (`torch.jit.script`), CVE-2025-3001 (`torch.lstm_cell`); local memory corruption | code paths this project does not call | not reachable | – |
| torch 2.9.1 | CVE-2026-4538 (no fix released): deserialization in the `pt2` loading handler, local | loading `torch.export` archives | not reachable | – |
| vllm 0.31.0 | CVE-2026-105922 (fixed in 0.32.0): denial of service through the penalty handler | a vLLM server reachable by an attacker | vLLM runs only for comparison stages, bound to `127.0.0.1`, behind an API key, inside a Pod whose only listener is key-only SSH; the load client always sends `repetition_penalty` 1 | upgrading means re-locking `requirements-vllm.txt` with a newer torch and re-running that stage; deferred, not forgotten |

Torch stays at 2.9.1 on purpose: it is the version of the production image and of the only measured GPU run (v0.9.1-4), and the production image was not to be modified. A torch upgrade
would also make the v1.0 numbers incomparable with that baseline. `requirements-service.txt` has no known advisories.

## Residual risks and what is not covered

- A Pod that is forgotten keeps billing, and a container exit is not assumed to stop it. This is a cost risk, not a confidentiality one; the runbook's termination checklist is the control.
- `artifacts.py` downloads `*.py` files from a model repository (never executed: `trust_remote_code` is not set). Model revisions are pinned by commit SHA and checked by the preflight.
- The image is built from a base pinned by digest, but system packages (`openssh-server`, `curl`, `tmux`, `procps`) are installed unpinned at build time. The image's own digest is what pins them for a given run.
- Runpod's platform, the SSH key you register, and the security of your workstation are outside this review.
