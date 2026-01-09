import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from setproctitle import *

setproctitle('AAAI26-LLM-LoRA')


base_path = "unseen ontology/vanila"
def inference(base_path):
    test_path = base_path+"/test_dataset.jsonl"
    lora_checkpoint = base_path+"/lora_llama3_0507"
    use_lora = True  # 🔁 LoRA on/off
    max_new_tokens = 120
    output_path = base_path+"/trained_outputs.jsonl" if use_lora else base_path+"/base_outputs.jsonl"
    # ========== model load ==========
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = base_model
    if use_lora:
        model = PeftModel.from_pretrained(base_model, lora_checkpoint)

    model.eval()

    # ========== data load ==========
    with open(test_path, "r") as f:
        data = [json.loads(line) for line in f]

    # ========== inference ==========
    results = []

    for example in tqdm(data):
        prompt = example["input"].strip()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        input_ids = inputs["input_ids"]  # 👈 실제 토큰 텐서를 꺼냄

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]  # ✅ 이제 OK
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "input": prompt,
            "generated_output": generated
        })

    # ========== save ==========

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in results:
            # 1. split by "|"
            parts = [part.strip() for part in ex["generated_output"].split("|") if "?" in part and part.strip()]
            
            # 2. transform to numbered dictionary
            numbered_output = {str(i): part for i, part in enumerate(parts)}
            
            # 3. save as JSON line
            json_line = json.dumps({
                "input": ex["input"],
                "generated_outputs": numbered_output
            }, ensure_ascii=False)
            
            f.write(json_line + "\n")
for path in ["unseen ontology/" + i for i in ["Pizza", "AWO", "DEM@Care", "OntoDT", "Stuff", "SWO"]]:
    inference(path)
