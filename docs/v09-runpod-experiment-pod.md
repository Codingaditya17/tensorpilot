# v0.9 experiment Pod: exact setup

The production worker image keeps `tensorpilot-server` as its entrypoint and contains no SSH, scripts
or benchmark configs. The experiment image adds those on top of the published production image. It is
a separate package, so the production image is untouched.

| | |
|---|---|
| Image | `ghcr.io/codingaditya17/tensorpilot/cuda-experiment:v0.9.1-3` |
| Built from | `ghcr.io/codingaditya17/tensorpilot/cuda@sha256:afdc5dc6646040cb486654cc6722a20f2d21a9c080f96ac5f33145a6a1da825a` (v0.9.1) |
| Published by | git tag `exp-v0.9.1-3` (workflow `experiment-image.yml`). Never moved: a fix gets `exp-v0.9.1-4`. |
| Registry login | none; the package must be public (check anonymously before renting) |

> Known issue in `v0.9.1-3`: the gateway smoke step fails (see the revision history). The next tag, `exp-v0.9.1-4`, is not published yet.

## Revision history

| Tag | Status |
|---|---|
| `v0.9.1-1` | broken: crashed when the volume refused `chown` |
| `v0.9.1-2` | broken: the preflight smoke failed on a real RTX 4090 with `ValueError: Expected a torch.device with a specified index` (bare `cuda` passed to `set_per_process_memory_fraction`) |
| `v0.9.1-3` | preflight and real GPU inference passed on an RTX 4090 (16 tokens), but the gateway smoke failed 5 of 5 with HTTP 401: the data plane had no worker credential. Do not use for the full run; the fix is committed and unpublished, pending review |

This image installs the repository's `tensorpilot` source over the base image's copy (and removes the base
image's `/app/tensorpilot`, which Python would load first from `/app`). The production image
`cuda:v0.9.1` was not modified and still contains the old backend, so it has the device bug until a production
release is cut deliberately.

Nothing here has run on a GPU successfully. The image behavior below was verified without one by
`deploy/runpod/test-pod-image.sh`.

## Runpod settings

| Setting | Value |
|---|---|
| Pod type | GPU Pod, 1 x RTX 4090 (24 GB) |
| Container image | the image above |
| Container disk | 40 GB |
| Volume disk | 20 GB, mount path `/workspace` |
| Expose HTTP ports | **none** |
| Expose TCP ports | `22` only |
| Container start command / entrypoint | **leave empty** (the image entrypoint starts sshd; overriding it skips SSH) |
| Environment variables | none required. `PUBLIC_KEY` is injected from your Runpod account SSH keys. If the container log says `PUBLIC_KEY is not set`, add your **public** key as a `PUBLIC_KEY` variable. Never put a private key or an API key in Runpod settings. |
| Network volume | none (the first run is disposable) |
| Jupyter / extra services | off |

The Pod volume is deleted with the Pod, so the evidence must be downloaded first (below).

## Storage: where evidence and the model cache go

At start the entrypoint picks each directory by testing it, not by trusting `chown`:
1. `/workspace/results` and `/workspace/huggingface` (the mounted volume) if the `tensorpilot` user can create,
   read back and delete a file there, trying `chown` only if that first test fails;
2. otherwise `/home/tensorpilot/work/...` on the **container disk**, with a `WARNING` in the container log;
3. otherwise the container exits with `no writable storage for ...` and a diagnostic (owner, mount, free space).

`v0.9.1-1` crashed on Runpod because it treated a refused `chown` on the volume as fatal. The log of a Pod that took
the fallback shows `storage: chown of /workspace/results refused: ...` followed by `WARNING results -> /home/...`.
Every SSH login prints `TensorPilot Pod: evidence -> <dir> | model cache -> <dir>`. **Use that path for the export.**
`first-run.sh`, `session.sh` and the cache all follow the selected directories.

If the fallback was used, evidence lives on the container disk, which does not survive a Pod stop or the 120-minute
backstop exit. Export it before either.

## Start (about 45 minutes, hard-capped)

The SSH user is `tensorpilot`, not `root`. Runpod's suggested command uses `root@`; change it. Use the
direct TCP address and port from the Pod's Connect tab (the proxied `ssh.runpod.io` route is not this sshd).

```bash
ssh -i ~/.ssh/<your-key> -p <public-port> tensorpilot@<public-ip>
tmux new -s run              # survives a dropped connection; reattach with: tmux attach -t run
nvidia-smi                   # confirm one RTX 4090; stop here if not
first-run.sh --plan          # generates three API keys in this shell, validates, starts nothing
first-run.sh                 # preflight, worker, gateway smoke, bounded benchmark, evidence archive
```

Caps: measured work is limited by `TENSORPILOT_BENCH_TIMEOUT=25m` and the three experiments' own limits;
the whole run is limited by `TENSORPILOT_SESSION_CAP=45m`. If a cap fires or any step fails, `session.sh`
archives the partial evidence before it exits. As a billing backstop the container exits after
`TENSORPILOT_POD_MAX_MINUTES=120`; an exited Pod stops GPU billing but is not terminated.

To look at the dashboard, tunnel it: `ssh -L 8080:127.0.0.1:8080 -p <port> tensorpilot@<ip>`. TensorPilot binds
127.0.0.1 only, and only port 22 listens publicly.

## Export the evidence (before terminating)

On the Pod, `first-run.sh` ends by printing the archive path and its SHA-256, and your login banner shows the
evidence directory (`/workspace/results` unless the fallback was used). From your laptop:

```bash
scp -i ~/.ssh/<your-key> -P <public-port> 'tensorpilot@<public-ip>:<evidence-dir>/*.tar.gz*' .
shasum -a 256 v09-*.tar.gz     # must equal the hash printed on the Pod
tar -tzf v09-*.tar.gz | head
```

The `.sha256` file records the Pod path, so compare the hash itself rather than using `shasum -c`.
Only after the hashes match is the evidence saved. If SSH is unreachable, the volume survives while the Pod is
stopped, but there is no second transfer path in this image; do not terminate until you have the archive.

## Then terminate

Follow section 9 of [`v09-runpod-runbook.md`](v09-runpod-runbook.md). Terminate the Pod (stopping keeps the volume billed).
