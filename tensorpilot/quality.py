"""Generation-quality fidelity: capture teacher-forced and greedy behavior, then compare two captures.

Quantization changes a model, so "does it still generate correctly" needs an appropriate reference. There are
three different questions and only the first two have a reference on a 24 GB card:

1. Is the *engine* correct on these weights? Batched decoding must equal the serial path on the same weights.
   That is tested in the repository (tests/test_cuda_batch*.py) and needs no reference model.
2. How much does quantization change a model whose full-precision version fits? Capture the bf16 model and its
   NF4 version (the 7B pair fits both ways) and compare them here.
3. How good is a model whose full-precision version does NOT fit (Qwen2.5-14B)? It cannot be compared against
   its own bf16 weights on this hardware. Its capture gives absolute numbers only (perplexity on a fixed text,
   greedy outputs) and the 7B comparison is the closest available evidence about NF4 fidelity, not a substitute.

    python -m tensorpilot.quality capture --model M --revision SHA --output ref.json
    python -m tensorpilot.quality capture --model M --revision SHA --quantization nf4 --reference ref.json --output nf4.json
    python -m tensorpilot.quality compare ref.json nf4.json

The built-in corpus is a few hundred words of public-domain text plus one technical paragraph. It is small and
its numbers are indicative; pass --corpus-file with a larger text for a stronger measurement.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA = "tensorpilot.quality/v1"
TOP_K = 20
MAX_SEQUENCE_TOKENS = 512  # bounds the [positions, vocabulary] logits tensor

BUILTIN_CORPUS = {
    "pride-and-prejudice": (
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in "
        "want of a wife. However little known the feelings or views of such a man may be on his first entering a "
        "neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is "
        "considered the rightful property of some one or other of their daughters. My dear Mr. Bennet, said his "
        "lady to him one day, have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had "
        "not."),
    "moby-dick": (
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, "
        "and nothing particular to interest me on shore, I thought I would sail about a little and see the watery "
        "part of the world. It is a way I have of driving off the spleen and regulating the circulation. "
        "Whenever I find myself growing grim about the mouth, whenever it is a damp, drizzly November in my soul, "
        "then I account it high time to get to sea as soon as I can."),
    "two-cities": (
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
        "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it "
        "was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything "
        "before us, we had nothing before us."),
    "declaration": (
        "When in the Course of human events, it becomes necessary for one people to dissolve the political bands "
        "which have connected them with another, and to assume among the powers of the earth, the separate and "
        "equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the "
        "opinions of mankind requires that they should declare the causes which impel them to the separation."),
    "alice": (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: "
        "once or twice she had peeped into the book her sister was reading, but it had no pictures or "
        "conversations in it, and what is the use of a book, thought Alice, without pictures or conversations?"),
    "inference": (
        "Autoregressive language model inference has two phases. Prefill processes the whole prompt in parallel and "
        "is limited by arithmetic throughput. Decode produces one token at a time and is limited by memory "
        "bandwidth, because every step reads all of the weights and the growing key and value cache. Batching "
        "several sequences into one forward pass amortizes the weight reads, which is why continuous batching "
        "raises throughput, while the cache grows with every sequence and bounds how many can run at once."),
}


def _corpus(path):
    if path:
        text = Path(path).read_text()
        paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
        return {f"paragraph-{index:04d}": block for index, block in enumerate(paragraphs)}
    return dict(BUILTIN_CORPUS)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def teacher_forced(model, token_ids, *, device, topk=TOP_K, reference_ids=None):
    """Next-token statistics at every position of ``token_ids``, computed in one forward pass."""
    import torch
    with torch.inference_mode():
        logits = model(torch.tensor([token_ids], device=device)).logits[0, :-1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        target = torch.tensor(token_ids[1:], device=device)
        chosen = logprobs.gather(1, target[:, None])[:, 0]
        best = logprobs.topk(min(topk, logprobs.shape[-1]), dim=-1)
        result = {"token_ids": list(token_ids), "logprob_of_next": [round(v, 6) for v in chosen.tolist()],
                  "top1": best.indices[:, 0].tolist(),
                  "topk_ids": best.indices.tolist(), "topk_logprobs": [[round(v, 6) for v in row]
                                                                         for row in best.values.tolist()]}
        if reference_ids is not None:
            # Exact log-probabilities at the reference run's top-k ids, so KL on that support needs no guess.
            index = torch.tensor(reference_ids, device=device)
            result["logprob_at_reference_topk"] = [[round(v, 6) for v in row]
                                                   for row in logprobs.gather(1, index).tolist()]
        return result


def capture(backend, *, corpus, prompts, generate_tokens, reference=None):
    tokenizer = backend.tokenizer
    sequences = {}
    for name, text in corpus.items():
        ids = tokenizer.encode(text)[:MAX_SEQUENCE_TOKENS]
        if len(ids) < 8:
            continue
        reference_ids = None
        if reference is not None:
            ref_entry = reference["teacher_forced"][name]
            if ref_entry["token_ids"] != ids:
                raise ValueError(f"{name}: the reference tokenized this text differently; tokenizers must match")
            reference_ids = ref_entry["topk_ids"]
        sequences[name] = teacher_forced(backend.model, ids, device=backend.device, reference_ids=reference_ids)
    generations = {}
    for index, prompt in enumerate(prompts):
        tokens = [token.token_id for token in backend.stream(prompt, generate_tokens) if token.output]
        generations[f"prompt-{index:02d}"] = {"prompt": prompt, "token_ids": tokens}
    provenance = backend.provenance
    return {"schema": SCHEMA, "model": provenance.get("requested_model") or None,
            "model_revision": provenance.get("model_revision"), "artifact_sha256": provenance.get("artifact_sha256"),
            "quantization": provenance.get("quantization"), "dtype": provenance.get("model_dtype"),
            "device": provenance.get("device"), "torch": provenance.get("torch_version"),
            "transformers": provenance.get("transformers_version"), "corpus_sha256": _fingerprint(corpus),
            "generate_tokens": generate_tokens, "teacher_forced": sequences, "generation": generations,
            "limitations": ["small fixed corpus: metrics are indicative, not a benchmark score",
                            "teacher-forced metrics measure next-token fidelity, not free-form answer quality"]}


def _mean(values):
    return sum(values) / len(values) if values else None


def compare(reference, test):
    """Metrics describing how far ``test`` departs from ``reference``. The reference is not assumed correct."""
    if reference.get("schema") != SCHEMA or test.get("schema") != SCHEMA:
        raise ValueError("both files must be tensorpilot.quality/v1 captures")
    if reference["corpus_sha256"] != test["corpus_sha256"]:
        raise ValueError("the captures used different corpora and cannot be compared")
    names = sorted(set(reference["teacher_forced"]) & set(test["teacher_forced"]))
    lp_ref, lp_test, agree, overlap, divergence, kl_values = [], [], [], [], [], []
    approximate_kl = False
    for name in names:
        r, t = reference["teacher_forced"][name], test["teacher_forced"][name]
        if r["token_ids"] != t["token_ids"]:
            raise ValueError(f"{name}: token ids differ; the two runs did not see the same text")
        lp_ref += r["logprob_of_next"]
        lp_test += t["logprob_of_next"]
        agree += [a == b for a, b in zip(r["top1"], t["top1"])]
        overlap += [len(set(a[:5]) & set(b[:5])) / 5 for a, b in zip(r["topk_ids"], t["topk_ids"])]
        for position, (ref_ids, ref_lp) in enumerate(zip(r["topk_ids"], r["topk_logprobs"])):
            # KL(reference || test) over the reference's top-k, renormalized on that support.
            mass = sum(math.exp(v) for v in ref_lp)
            p = [math.exp(v) / mass for v in ref_lp]
            if "logprob_at_reference_topk" in t:
                q_lp = t["logprob_at_reference_topk"][position]
            else:
                approximate_kl = True
                known = dict(zip(t["topk_ids"][position], t["topk_logprobs"][position]))
                floor = min(t["topk_logprobs"][position]) - math.log(2)
                q_lp = [known.get(i, floor) for i in ref_ids]
            q_mass = sum(math.exp(v) for v in q_lp)
            q = [math.exp(v) / q_mass for v in q_lp]
            kl_values.append(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0))
    ppl_ref, ppl_test = (math.exp(-_mean(v)) for v in (lp_ref, lp_test)) if lp_ref else (None, None)
    generation_names = sorted(set(reference["generation"]) & set(test["generation"]))
    exact, prefix = 0, []
    for name in generation_names:
        a, b = reference["generation"][name]["token_ids"], test["generation"][name]["token_ids"]
        shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        exact += a == b
        prefix.append(shared / max(len(a), len(b), 1))
        if a != b:
            divergence.append(shared)
    return {"schema": "tensorpilot.quality-comparison/v1",
            "reference": {k: reference.get(k) for k in ("model", "model_revision", "quantization", "dtype")},
            "test": {k: test.get(k) for k in ("model", "model_revision", "quantization", "dtype")},
            "positions_compared": len(lp_ref), "sequences_compared": len(names),
            "perplexity_reference": ppl_ref, "perplexity_test": ppl_test,
            "perplexity_ratio_test_over_reference": ppl_test / ppl_ref if ppl_ref else None,
            "top1_agreement": _mean(agree), "top5_overlap": _mean(overlap),
            "kl_reference_to_test_nats_topk_support": _mean(kl_values),
            "kl_is_approximate": approximate_kl,
            "generations_compared": len(generation_names),
            "greedy_exact_match_fraction": exact / len(generation_names) if generation_names else None,
            "greedy_mean_prefix_agreement": _mean(prefix),
            "greedy_first_divergence_index_mean": _mean(divergence),
            "reading_guide": ["a ratio near 1.0 and agreement near 1.0 mean the test model behaves like the reference",
                              "greedy outputs diverge at the first near-tie, so exact match falls quickly with length; "
                              "the prefix agreement and the teacher-forced metrics are more informative",
                              "this compares two models; it does not say either one answers questions correctly"]}


def _load_backend(args):
    from .cuda_backend import CUDABackend
    return CUDABackend(args.model, revision=args.revision, device=args.device, dtype=args.dtype,
                       cache_dir=args.cache_dir, offline=args.offline, quantization=args.quantization,
                       allow_cpu=args.allow_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture", help="run a model over the corpus and prompts and save its behavior")
    cap.add_argument("--model", required=True)
    cap.add_argument("--revision")
    cap.add_argument("--device", default="cuda")
    cap.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    cap.add_argument("--quantization", default="none", choices=("none", "nf4"))
    cap.add_argument("--cache-dir")
    cap.add_argument("--offline", action="store_true")
    cap.add_argument("--allow-cpu", action="store_true", help="for local testing only")
    cap.add_argument("--corpus-file", help="text file; paragraphs separated by blank lines")
    cap.add_argument("--reference", help="a capture to gather exact log-probabilities against")
    cap.add_argument("--generate-tokens", type=int, default=48)
    cap.add_argument("--output", required=True)
    cmp = commands.add_parser("compare", help="compare two captures")
    cmp.add_argument("reference")
    cmp.add_argument("test")
    cmp.add_argument("--output")
    args = parser.parse_args()
    if args.command == "capture":
        from .bench import PROMPTS
        reference = json.loads(Path(args.reference).read_text()) if args.reference else None
        document = capture(_load_backend(args), corpus=_corpus(args.corpus_file), prompts=PROMPTS,
                           generate_tokens=args.generate_tokens, reference=reference)
        document["model"] = args.model
        Path(args.output).write_text(json.dumps(document) + "\n")
        print(f"saved {args.output}: {len(document['teacher_forced'])} sequences, "
              f"{len(document['generation'])} generations")
        return 0
    report = compare(json.loads(Path(args.reference).read_text()), json.loads(Path(args.test).read_text()))
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
