"""
Inference utilities with KV Cache for efficient autoregressive generation.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('inf'),
) -> torch.Tensor:
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: [batch, vocab_size]
        top_k: Keep only top k tokens
        top_p: Keep tokens with cumulative probability >= top_p
        
    Returns:
        Filtered logits
    """
    if top_k > 0:
        # Remove tokens with probability less than the top_k
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    if top_p < 1.0:
        # Sort logits and compute cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    return logits


@torch.no_grad()
def generate_with_cache(model, input_ids, config, tokenizer=None):
    """Generate with real caches, or recompute the full prefix when unavailable.

    Models return either (logits, cache) or an object with logits and optional
    past_key_values. A missing cache always means full-prefix recomputation.
    """
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError('input_ids must contain a nonempty prompt per row')
    if config.max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    if config.do_sample and config.temperature <= 0:
        raise ValueError('sampling temperature must be positive')
    generated = input_ids.clone()
    if config.max_new_tokens == 0:
        return generated

    def unpack(outputs):
        if isinstance(outputs, tuple):
            return outputs[0], outputs[1] if len(outputs) > 1 else None
        return (getattr(outputs, 'logits', outputs),
                getattr(outputs, 'past_key_values', None))

    def has_cache(cache):
        if cache is None:
            return False
        if isinstance(cache, (list, tuple)):
            return len(cache) > 0
        if hasattr(cache, 'get_seq_length'):
            return cache.get_seq_length() > 0
        return True

    logits, caches = unpack(model(generated, use_cache=True))
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for step in range(config.max_new_tokens):
        next_logits = logits[:, -1, :]
        if config.do_sample:
            filtered = top_k_top_p_filtering(
                next_logits / config.temperature, config.top_k, config.top_p
            )
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), 1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        if config.eos_token_id is not None:
            pad_id = config.pad_token_id
            if pad_id is None:
                pad_id = config.eos_token_id
            next_token = torch.where(finished[:, None], pad_id, next_token)
            finished = finished | (next_token[:, 0] == config.eos_token_id)
        generated = torch.cat((generated, next_token), dim=-1)
        if finished.all() or step + 1 == config.max_new_tokens:
            break
        if has_cache(caches):
            outputs = model(next_token, past_key_values=caches, use_cache=True)
        else:
            outputs = model(generated, use_cache=True)
        logits, caches = unpack(outputs)
    return generated


class InferenceEngine:
    """
    High-level inference engine with caching for SafeMoE models.
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        model.to(device)
        model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text including prompt
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Configure
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        output_ids = generate_with_cache(self.model, input_ids, config)
        
        # Decode
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate for multiple prompts."""
        return [self.generate(p, **kwargs) for p in prompts]

# Alias
class SafeMoEGenerator(InferenceEngine):
    """Alias for InferenceEngine matching SafeMoE export style."""
    pass
