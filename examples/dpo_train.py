from trl import DPOTrainer, DPOConfig

# policy 模型（要训练的）
policy = AutoModelForCausalLM.from_pretrained(
    "./sft_lora",  # 或直接加载 base，再加载 adapter
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# reference 模型（冻结；通常用 base 或 SFT 前的 checkpoint）
ref = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

dpo_args = DPOConfig(
    output_dir="./dpo_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    beta=0.1,
    max_length=4096,
    max_prompt_length=2048,
)

dpo_trainer = DPOTrainer(
    model=policy,
    ref_model=ref,
    args=dpo_args,
    train_dataset=dpo_dataset,  # 需包含 prompt/chosen/rejected
    tokenizer=tokenizer,
    peft_config=lora_config_dpo,
)
dpo_trainer.train()
dpo_trainer.save_model()
