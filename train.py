from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load datasets
hermes = load_dataset("NousResearch/hermes-function-calling-v1", split="train")

with open("data/custom_dataset.json", "r") as f:
    custom_data = json.load(f)

hermes_formatted = [{"conversations": ex["conversations"]} for ex in hermes]
combined = hermes_formatted + custom_data
dataset = Dataset.from_list(combined)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="conversations",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)

trainer.train()
model.save_pretrained("toolchain-qwen2.5-3b")
