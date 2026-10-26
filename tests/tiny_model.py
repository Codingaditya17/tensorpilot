"""Offline, randomly initialized transformer for CPU tests of the CUDA backend.

This is not a language model and carries no quality claim. It exercises the same
Transformers loading, KV-cache and tokenizer code paths as a real checkpoint, with
no network access and no GPU.
"""
from pathlib import Path

WORDS = ["<unk>", "</s>"] + [f"w{index}" for index in range(60)] + ["hello", "world", "the", "a"]
EOS_ID = 1


def make_tiny_model_dir(directory, *, seed=0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    directory = Path(directory)
    vocab = {word: index for index, word in enumerate(WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="</s>")
    # A Llama-type config makes AutoTokenizer load the tokenizer.json written below. A
    # Qwen-type config would select a byte-level BPE class that needs vocab/merges files.
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, eos_token_id=EOS_ID, bos_token_id=None,
                         pad_token_id=None)
    torch.manual_seed(seed)
    LlamaForCausalLM(config).save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


def make_tiny_qwen2_model(*, seed=1):
    """An in-memory randomly initialized Qwen2 model with the same vocabulary as the tokenizer fixture.

    A Qwen2 config makes AutoTokenizer select a byte-level BPE class that cannot load the word-level
    tokenizer, so tests keep the Llama directory for tokenization and swap this model in. Qwen2 has biased
    q/k/v projections, the layout of the real Qwen2.5 checkpoints.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         eos_token_id=EOS_ID, bos_token_id=None, pad_token_id=None)
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(config).eval()
