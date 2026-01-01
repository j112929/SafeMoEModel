from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# 1) config/model
cfg = SafeMoETransformerConfig(
    vocab_size=50257,
    max_position_embeddings=2048,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    moe_num_experts=8,
    moe_top_k=2,
    moe_capacity_factor=1.25,
    moe_route_threshold=0.05,
)
model = SafeMoETransformerLM(cfg)

# 2) data collator (causal LM)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3) trainer
args = TrainingArguments(
    output_dir="out-safe-moe",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    num_train_epochs=1,
    logging_steps=20,
    save_steps=500,
    bf16=True,   # 有条件建议 bf16
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)
trainer.train()
