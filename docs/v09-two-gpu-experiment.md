# v0.9 two-GPU experiment (prepared, not executed)

Run only after the single-GPU session passed and you have chosen to rent a Pod with two or
more GPUs. Nothing here has run on a GPU; `deploy/runpod/two_gpu_session.sh --plan` is the only
mode exercised.

## What it can and cannot show

Two worker processes, each pinned to one GPU with `CUDA_VISIBLE_DEVICES`, behind the control
plane and data plane. This is two independent GPU processes on **one host**. Two GPUs in one
Pod are not two nodes: the workers talk over loopback, so the experiment says nothing about
inter-node networking. The script writes `placement.json` (hostname, `/etc/machine-id`, GPU
UUIDs, PCI bus ids). A report may say "multi-node" only if separate Pods are used and their
hostnames, machine ids and network path are recorded.

## Phases (same load in each: 60 requests, concurrency 8, 64 tokens, seed 7)

| Phase | Target | Question |
| --- | --- | --- |
| A | worker-a direct | single-GPU reference measured in the same session |
| B | gateway, 2 workers | load distribution, per-GPU memory and utilization, throughput and latency vs A |
| C | gateway, `kill -9` worker-b mid-load | detection time, request outcomes (one terminal event each, no replayed tokens) |
| D | gateway after restart | worker re-registers as available, load succeeds |

Evidence per run: `placement.json`, `gpu-samples.csv` (both GPUs, 1 s), `workers.*.json`
registry snapshots, `worker-{a,b}.metrics.*.txt`, request journals, `failure-injection.json`
(kill timestamp), worker/control/data logs, checksum.

## Honest expectations

CUDA execution is serial per worker. Concurrency 8 over one worker queues; over two workers it
queues half as deep. Expect throughput near 2x only if the workload is queue-bound. Report
that as capacity scaling by routing, not as batching. Model-aware placement is not exercised:
both workers serve the same model. Testing it needs a second model and is left out on purpose.

## Cost shape

Roughly 1 to 1.5 hours on the Pod, so 2 to 3 GPU-hours, plus model download on first start.
A planning estimate, not a measurement. Terminate the Pod when the archive is downloaded.
