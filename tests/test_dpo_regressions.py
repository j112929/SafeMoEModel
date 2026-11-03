import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformers')
pytest.importorskip('datasets')
from examples.dpo_demo import DPOTrainer, tokenize_preference_pair


class Tokenizer:
    pad_token_id = 31
    eos_token_id = 31

    def encode(self, text, add_special_tokens=False):
        return [len(word) for word in text.split()]


def test_completion_likelihood_is_a_masked_sum_with_safe_indices():
    class UniformModel:
        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 7))
    ids = torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]])
    labels = torch.tensor([[-100, 2, 3, -100], [-100, 4, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    actual = DPOTrainer._get_logprobs(None, UniformModel(), ids, labels, mask)
    torch.testing.assert_close(actual, torch.tensor([-2 * math.log(7), -math.log(7)]))


def test_prompt_and_padding_are_masked_but_real_eos_is_a_target():
    record = tokenize_preference_pair(Tokenizer(), {
        'prompt': 'a prompt', 'chosen': 'a longer answer', 'rejected': 'no',
    }, 8)
    for side in ['chosen', 'rejected']:
        ids, labels, mask = [record[side + '_' + key] for key in ['input_ids', 'labels', 'attention_mask']]
        assert labels[:2] == [-100, -100]
        assert ids[sum(mask) - 1] == labels[sum(mask) - 1] == 31
        assert all(label == -100 for label, valid in zip(labels, mask) if not valid)
    assert sum(record['chosen_attention_mask']) != sum(record['rejected_attention_mask'])


def test_dpo_trainer_takes_a_real_optimizer_step(tmp_path):
    from datasets import Dataset
    from transformers import TrainingArguments
    from hf_integration.safe_moe_hf import SafeMoETransformerConfig, SafeMoETransformerLM
    model = SafeMoETransformerLM(SafeMoETransformerConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, moe_num_experts=2,
    ))
    before = model.lm_head.weight.detach().clone()
    row = tokenize_preference_pair(Tokenizer(), {
        'prompt': 'a prompt', 'chosen': 'a good answer', 'rejected': 'no',
    }, 8)
    trainer = DPOTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=2,
        use_cpu=True, report_to='none', save_strategy='no', disable_tqdm=True,
        remove_unused_columns=False, dataloader_pin_memory=False,
    ), train_dataset=Dataset.from_list([row, row]))
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    assert not torch.equal(before, model.lm_head.weight)
    assert all(parameter.grad is None for parameter in trainer.ref_model.parameters())
