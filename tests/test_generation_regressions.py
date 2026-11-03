from types import SimpleNamespace

import pytest
import torch
from torch import nn

from safemoe import GenerationConfig, generate_with_cache


class PrefixModel(nn.Module):
    def __init__(self, cached=False):
        super().__init__()
        self.cached = cached
        self.lengths = []

    def forward(self, input_ids, use_cache=True, past_key_values=None):
        self.lengths.append(input_ids.size(1))
        prefix = input_ids if past_key_values is None else torch.cat((past_key_values, input_ids), 1)
        # The next token depends on every token in the prefix.
        prediction = prefix.sum(-1) % 7
        logits = torch.nn.functional.one_hot(prediction, 7).float()[:, None, :]
        return SimpleNamespace(logits=logits, past_key_values=prefix if self.cached else None)


def test_cache_and_full_prefix_paths_generate_the_same_history():
    prompt = torch.tensor([[1, 2]])
    cfg = GenerationConfig(max_new_tokens=4, do_sample=False, temperature=0)
    cached, uncached = PrefixModel(True), PrefixModel(False)
    expected = torch.tensor([[1, 2, 3, 6, 5, 3]])
    torch.testing.assert_close(generate_with_cache(cached, prompt, cfg), expected)
    torch.testing.assert_close(generate_with_cache(uncached, prompt, cfg), expected)
    assert cached.lengths == [2, 1, 1, 1]
    assert uncached.lengths == [2, 3, 4, 5]


def test_hf_model_generation_matches_manual_full_prefix():
    pytest.importorskip('transformers')
    from hf_integration.safe_moe_hf import SafeMoETransformerConfig, SafeMoETransformerLM
    torch.manual_seed(8)
    model = SafeMoETransformerLM(SafeMoETransformerConfig(
        vocab_size=11, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, moe_num_experts=2,
    )).eval()
    prompt = torch.tensor([[1, 2, 3]])
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(4):
            token = model(expected).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), 1)
    actual = generate_with_cache(model, prompt, GenerationConfig(max_new_tokens=4, do_sample=False))
    torch.testing.assert_close(actual, expected)


def test_finished_rows_stay_finished():
    class EOSModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, ids, **kwargs):
            choices = [[0, 1], [1, 0]][self.calls]
            self.calls += 1
            return SimpleNamespace(logits=torch.nn.functional.one_hot(torch.tensor(choices), 3).float()[:, None])
    actual = generate_with_cache(EOSModel(), torch.tensor([[2], [2]]), GenerationConfig(
        max_new_tokens=5, do_sample=False, eos_token_id=0, pad_token_id=2,
    ))
    torch.testing.assert_close(actual, torch.tensor([[2, 0, 2], [2, 1, 0]]))


def test_sampling_handles_a_small_vocabulary_and_rejects_zero_temperature():
    prompt = torch.tensor([[1, 2]])
    actual = generate_with_cache(PrefixModel(), prompt, GenerationConfig(max_new_tokens=1))
    assert actual.shape == (1, 3)  # Default top_k=50 exceeds this model's vocabulary.
    with pytest.raises(ValueError, match='temperature'):
        generate_with_cache(PrefixModel(), prompt, GenerationConfig(temperature=0))
