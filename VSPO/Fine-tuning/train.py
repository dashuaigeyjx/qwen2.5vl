from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model
from setproctitle import *
from datetime import datetime
import os
setproctitle('AAAI26-LLM-LoRA')
def train_lora_model(dataset_dir):

    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_dir = f"{dataset_dir}/lora_llama3_0507"


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=[],
    )

    model = get_peft_model(model, peft_config)


    dataset = load_dataset("json", data_files={
        "train": f"{dataset_dir}/train_dataset.jsonl",
        "test": f"{dataset_dir}/test_dataset.jsonl"
    })

    # Preprocessing
    def preprocess(example):
        prompt = example["input"].strip()
        output = example["output"].strip()
        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=768, padding=False)
        prompt_len = len(prompt_tokenized["input_ids"])
        full_tokenized = tokenizer(prompt + output, truncation=True, max_length=768, padding="max_length")
        labels = [-100] * prompt_len + full_tokenized["input_ids"][prompt_len:]
        labels = labels[:768]
        labels += [-100] * (768 - len(labels))
        full_tokenized["labels"] = labels
        return full_tokenized

    train_dataset = dataset["train"].map(preprocess, batched=False)
    eval_dataset = dataset["test"].map(preprocess, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./{model_dir}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        max_steps=-1,
        logging_dir=f"./{model_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(f"./{model_dir}")

# ─────────────────────────────────────
# Path to the data selection directory
data_selection_dir = 'unseen ontology'
# Get the list of directories
directories = [data_selection_dir+'/'+d for d in os.listdir(data_selection_dir) if os.path.isdir(os.path.join(data_selection_dir, d))]
print(directories)

for dataset_dir in directories:
    train_lora_model(dataset_dir)
