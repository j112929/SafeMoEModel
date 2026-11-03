"""
DPO (Direct Preference Optimization) Training Demo for SafeMoE
Uses Hugging Face Trainer with preference pairs.
"""
import os
import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
import torch.nn.functional as F

# Import local HF model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration.safe_moe_hf import SafeMoETransformerLM, SafeMoETransformerConfig


class DPOTrainer(Trainer):
    """
    Simple DPO Trainer that computes preference loss.
    
    DPO Loss = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) 
                                    - log_ref(chosen) + log_ref(rejected))))
    
    For this demo, we use the model itself as the reference (frozen copy).
    """
    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        # Create a frozen reference copy
        self.ref_model = None  # Will copy on first forward
        
    def _get_logprobs(self, model, input_ids, labels, attention_mask=None):
        """Sum completion log likelihoods, ignoring prompt/padding labels."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if attention_mask is not None:
            mask = mask & attention_mask[:, 1:].bool()
        safe_labels = shift_labels.masked_fill(~mask, 0)
        logps = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
        gathered = logps.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return gathered.masked_fill(~mask, 0).sum(dim=-1)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute DPO loss."""
        # Initialize reference model on first call
        if self.ref_model is None:
            import copy
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        
        # Get inputs
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        chosen_mask = inputs.get("chosen_attention_mask")
        rejected_labels = inputs["rejected_labels"]
        rejected_mask = inputs.get("rejected_attention_mask")
        
        # Policy log probs
        pi_chosen = self._get_logprobs(model, chosen_ids, chosen_labels, chosen_mask)
        pi_rejected = self._get_logprobs(model, rejected_ids, rejected_labels, rejected_mask)
        
        # Reference log probs
        with torch.no_grad():
            ref_chosen = self._get_logprobs(self.ref_model, chosen_ids, chosen_labels, chosen_mask)
            ref_rejected = self._get_logprobs(self.ref_model, rejected_ids, rejected_labels, rejected_mask)
        
        # DPO loss
        pi_diff = pi_chosen - pi_rejected
        ref_diff = ref_chosen - ref_rejected
        
        loss = -F.logsigmoid(self.beta * (pi_diff - ref_diff)).mean()
        
        if return_outputs:
            return loss, {"pi_diff": pi_diff.mean().item(), "ref_diff": ref_diff.mean().item()}
        return loss



def tokenize_preference_pair(tokenizer, pair, max_length):
    """Build right-padded completion-only labels, preserving real EOS targets.

    Tokenize the prompt and response separately so BPE boundary merges cannot
    accidentally place response tokens inside the ignored prompt span.
    """
    if max_length < 3:
        raise ValueError('max_length must leave room for a prompt and completion')
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError('a pad_token_id is required')
    prompt_ids = tokenizer.encode(pair['prompt'], add_special_tokens=False)
    if not prompt_ids:
        raise ValueError('preference prompts must contain at least one token')
    prompt_ids = prompt_ids[:max_length - 2]
    result = {}
    for side in ('chosen', 'rejected'):
        completion = tokenizer.encode(' ' + pair[side], add_special_tokens=False)
        room = max_length - len(prompt_ids)
        eos = tokenizer.eos_token_id
        if eos is not None:
            if completion and completion[-1] == eos:
                completion = completion[:-1]
            completion = completion[:room - 1] + [eos]
        else:
            completion = completion[:room]
        if not completion:
            raise ValueError('preference completions must contain a target token')
        input_ids = prompt_ids + completion
        padding = max_length - len(input_ids)
        result[side + '_input_ids'] = input_ids + [pad_id] * padding
        result[side + '_attention_mask'] = [1] * len(input_ids) + [0] * padding
        result[side + '_labels'] = ([-100] * len(prompt_ids) + completion
                                   + [-100] * padding)
    return result


def create_preference_dataset(tokenizer, max_length: int = 64):
    """Create a dummy preference dataset."""
    preference_pairs = [
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of AI that enables systems to learn from data.",
            "rejected": "I don't know."
        },
        {
            "prompt": "Explain MoE models.",
            "chosen": "Mixture of Experts models use sparse activation, routing tokens to specialized sub-networks.",
            "rejected": "MoE is something about experts."
        },
        {
            "prompt": "How to write clean code?",
            "chosen": "Use meaningful names, keep functions small, write tests, and refactor regularly.",
            "rejected": "Just write code that works."
        },
    ] * 20

    def tokenize_pair(pair):
        return tokenize_preference_pair(tokenizer, pair, max_length)

    processed = [tokenize_pair(p) for p in preference_pairs]
    return Dataset.from_list(processed)


def run_dpo_demo():
    print("=== Starting DPO Demo with SafeMoE ===")
    
    # 1. Config
    print("Initializing Config...")
    config = SafeMoETransformerConfig(
        vocab_size=50257,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        moe_num_experts=4,
        moe_top_k=2,
    )
    
    # 2. Model
    print("Initializing Model...")
    model = SafeMoETransformerLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Dataset
    print("Creating Preference Dataset...")
    dataset = create_preference_dataset(tokenizer)
    
    # 5. Training Args
    args = TrainingArguments(
        output_dir="./dpo_output",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=5,
        save_steps=1000,
        learning_rate=5e-5,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        save_safetensors=False
    )
    
    # 6. DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        beta=0.1,
    )
    
    # 7. Train
    print("Starting DPO Training...")
    trainer.train()
    print("DPO Training Completed Successfully!")


if __name__ == "__main__":
    run_dpo_demo()
