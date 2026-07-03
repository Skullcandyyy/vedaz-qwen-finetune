"""Debug script to catch exact training error."""
import json, os, sys, torch, traceback
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

DATA_PATH = r"C:\Users\ACER-PC\Downloads\Chat Data for assessment of applicants (1).json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = r"C:\Users\ACER-PC\vedaz-finetune\qwen-vedaz-lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # ── Load data ──────────────────────────────────────
    print("Loading data...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    idx, raw = 0, []
    while True:
        stripped = content[idx:].lstrip()
        if not stripped:
            break
        try:
            obj, idx2 = decoder.raw_decode(stripped)
            raw.append(obj)
            idx = content.find(stripped, idx) + idx2
        except json.JSONDecodeError:
            idx += 1

    conversations = [item["messages"] for item in raw
                     if isinstance(item, dict) and "messages" in item]
    print(f"Loaded {len(conversations)} conversations")

    # ── Tokenizer ──────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Format & tokenize ──────────────────────────────
    MAX_LEN = 512

    def format_and_mask(messages):
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(full_text, truncation=True, max_length=MAX_LEN)
        input_ids = enc["input_ids"]

        boundaries = [0]
        for i in range(1, len(messages) + 1):
            t = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=False
            )
            tok = tokenizer(t, truncation=True, max_length=MAX_LEN)
            boundaries.append(min(len(tok["input_ids"]), MAX_LEN))

        labels = [-100] * len(input_ids)
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                for j in range(boundaries[i], min(boundaries[i + 1], len(labels))):
                    labels[j] = input_ids[j]

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": [1] * len(input_ids),
        }

    print("Tokenizing dataset...")
    dataset_list = [format_and_mask(c) for c in conversations]
    dataset = Dataset.from_list(dataset_list)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Model ─────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────
    print("Setting up trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,       # reduced for CPU stability
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=2,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
        optim="adamw_torch",
        use_cpu=True,          # explicitly use CPU (transformers 5.x)
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SUCCESS! Model saved to {OUTPUT_DIR}")

except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Type   : {type(e).__name__}")
    print(f"Message: {e}")
    print(f"\nTraceback:")
    traceback.print_exc()
    sys.exit(1)
