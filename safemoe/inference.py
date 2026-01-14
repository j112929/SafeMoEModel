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
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
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
def generate_with_cache(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    tokenizer=None,
) -> torch.Tensor:
    """
    Generate text using KV cache for efficient autoregressive generation.
    
    This is a reference implementation. For production, use model-specific
    generate() methods that handle edge cases better.
    
    Args:
        model: Model with forward(input_ids, use_cache=True) -> (logits, caches)
        input_ids: [batch, seq_len] initial tokens
        config: Generation configuration
        tokenizer: Optional tokenizer for decoding (for debugging)
        
    Returns:
        generated_ids: [batch, seq_len + new_tokens]
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    # Track generated tokens
    generated = input_ids.clone()
    
    # Process initial prompt (prefill)
    # Note: This assumes model returns (logits, list_of_kv_caches)
    # You may need to adapt this to your model's interface
    outputs = model(input_ids, use_cache=True)
    
    # Handle different output formats
    if isinstance(outputs, tuple):
        logits = outputs[0]  # [B, T, V]
        caches = outputs[1] if len(outputs) > 1 else None
    else:
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        caches = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
    
    # Get logits for last position
    next_token_logits = logits[:, -1, :]  # [B, V]
    
    for step in range(config.max_new_tokens):
        # Apply temperature
        if config.temperature != 1.0:
            next_token_logits = next_token_logits / config.temperature
        
        # Apply filtering
        if config.do_sample:
            filtered_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=config.top_k,
                top_p=config.top_p
            )
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append to generated
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Check for EOS
        if config.eos_token_id is not None:
            if (next_token == config.eos_token_id).all():
                break
        
        # Forward with cache (only new token)
        outputs = model(next_token, past_key_values=caches, use_cache=True)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
            caches = outputs[1] if len(outputs) > 1 else caches
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            caches = outputs.past_key_values if hasattr(outputs, 'past_key_values') else caches
        
        next_token_logits = logits[:, -1, :]
    
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
