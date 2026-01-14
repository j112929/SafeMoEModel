
import os
import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

# Import local HF model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration.safe_moe_hf import SafeMoETransformerLM, SafeMoETransformerConfig

def run_sft_demo():
    print("=== Starting SFT Demo with SafeMoE (Standard Trainer) ===")
    
    # 1. Config
    print("Initializing Config...")
    config = SafeMoETransformerConfig(
        vocab_size=50257, # GPT2 size
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        moe_num_experts=4,
        moe_top_k=2,
        moe_min_capacity=2
    )
    
    # 2. Model
    print("Initializing Model...")
    model = SafeMoETransformerLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Tokenizer
    print("Initializing Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("Using dummy tokenizer")
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=None, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
        
    # 4. Dataset
    print("Creating Dataset...")
    texts = [
        "Hello, how are you? I am a SafeMoE model.",
        "Refactoring code is important for maintainability.",
        "Machine learning requires good data and architecture.",
        "Mixture of Experts allows sparse activation.",
    ] * 20
    
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 5. Training Args
    print("Setting up Trainer...")
    args = TrainingArguments(
        output_dir="./sft_output",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=5,
        save_steps=1000,
        learning_rate=1e-4,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False, # Important for custom models sometimes
        save_safetensors=False       # Fixes tied weights saving issue
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    # 7. Train
    print("Starting Training...")
    trainer.train()
    print("SFT Training Completed Successfully!")

if __name__ == "__main__":
    run_sft_demo()
