"""CPU conformance stubs only; no real batching performance claim."""
import threading
from types import SimpleNamespace

import pytest

from tensorpilot.batch_backend import MLXBatchAdapter


class Detokenizer:
    def __init__(self):
        self.segment = ""

    def add_token(self, token):
        self.segment += str(token)

    def finalize(self):
        pass

    @property
    def last_segment(self):
        text, self.segment = self.segment, ""
        return text


class Tokenizer:
    eos_token_ids = {99}

    def encode(self, prompt):
        return list(range(len(prompt)))

    @property
    def detokenizer(self):
        return Detokenizer()


class Generator:
    def __init__(self, model, **kwargs):
        self.pending = {}
        self.next_uid = 0
        self.closed = False

    def insert(self, prompts, max_tokens):
        uid = self.next_uid
        self.next_uid += 1
        self.pending[uid] = max_tokens[0]
        return [uid]

    def next(self):
        responses = [SimpleNamespace(uid=uid, token=1, finish_reason="length") for uid in self.pending]
        self.pending.clear()
        return [], responses

    def remove(self, uids):
        for uid in uids:
            self.pending.pop(uid, None)

    def close(self):
        self.closed = True


def adapter(**kwargs):
    backend = SimpleNamespace(model=None, tokenizer=Tokenizer(), max_prompt_tokens=10)
    return MLXBatchAdapter(backend, generator_factory=Generator, **kwargs)


def test_sequence_state_and_budgets_are_independent():
    batch = adapter(capacity=2)
    first, second = batch.insert("abc", 1), batch.insert("abcdef", 2)
    assert batch.sequences[first].detokenizer is not batch.sequences[second].detokenizer
    assert batch.generator.pending == {first: 1, second: 2}
    events = dict(batch.step())
    assert events[first].text == events[second].text == "1"
    assert events[first].prompt_tokens == 3
    assert events[second].prompt_tokens == 6
    assert not batch.sequences
    batch.close()


def test_cancellation_removes_only_one_sequence_and_can_be_repeated():
    batch = adapter()
    first, second = batch.insert("a", 1), batch.insert("b", 1)
    assert batch.remove(first)
    assert not batch.remove(first)
    assert [uid for uid, _ in batch.step()] == [second]
    batch.close()
    batch.close()


def test_capacity_and_prompt_limits_do_not_mutate_valid_sequences():
    batch = adapter(capacity=1)
    with pytest.raises(ValueError):
        batch.insert("x" * 11, 1)
    first = batch.insert("a", 1)
    with pytest.raises(RuntimeError):
        batch.insert("b", 1)
    assert list(batch.sequences) == [first]
    batch.close()


def test_cross_thread_calls_are_rejected():
    batch = adapter()
    errors = []

    def other_thread():
        try:
            batch.step()
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    assert "model-owner thread" in errors[0]
    batch.close()
