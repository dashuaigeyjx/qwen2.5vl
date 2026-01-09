import json

# ========== setting ==========
input_file = "GPT_4_1_output.jsonl"   # GPT output batch file
output_file = "parsed_GPT_4_1_outputs.jsonl"    # output file

# ========== processing ==========
results = []

# 1. read JSONL file
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        custom_id = data["custom_id"]
        content = data["response"]["body"]["choices"][0]["message"]["content"]

        # 2. split by "|" and clean up
        parts = [part.strip() for part in content.split("|") if "?" in part and part.strip()]
        numbered_output = {str(i): part for i, part in enumerate(parts)}

        # 3. save to results
        results.append({
            "custom_id": custom_id,
            "generated_outputs": numbered_output
        })

# 4. save to JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ parsed : {output_file}")
