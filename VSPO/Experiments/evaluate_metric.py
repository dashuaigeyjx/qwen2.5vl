from evaluation import eval_result
from evaluation import analyze_tcq_relation_metrics
import json

base_model_path = "vanila/base_outputs.jsonl"
trained_model_path = "vanila/trained_outputs.jsonl"

with open(base_model_path, 'r') as f:
    base_model_outputs = [json.loads(line) for line in f]

with open(trained_model_path, 'r') as f:
    trained_model_outputs = [json.loads(line) for line in f]

GPT4_1_path = "GPT/parsed_GPT_4_1_outputs.jsonl"
gold_path = "test_dataset_meta.jsonl"

with open(GPT4_1_path, 'r') as f:
    GPT4_1_outputs = [json.loads(line) for line in f]

with open(gold_path, 'r') as f:
    Gold = [json.loads(line) for line in f]


with open("GPT_result.json", "w") as outfile:
    json.dump(eval_result(GPT4_1_outputs, Gold), outfile, indent=4, ensure_ascii=False)
with open("base_result.json", "w") as outfile:
    json.dump(eval_result(base_model_outputs, Gold), outfile, indent=4, ensure_ascii=False)
with open("trained_result.json", "w") as outfile:
    json.dump(eval_result(trained_model_outputs, Gold), outfile, indent=4, ensure_ascii=False)

# unseen ontology experiment
unseen_path = "unseen ontology/Pizza/trained_outputs.jsonl" 
unseen_gold_path = "unseen ontology/Pizza/test_dataset_meta.jsonl"
with open(unseen_path, 'r') as f:
    unseen_gen_outputs = [json.loads(line) for line in f]
with open(unseen_gold_path, 'r') as f:
    unseen_Gold = [json.loads(line) for line in f]
with open("unseen_Pizza_result.json", "w") as outfile:
    json.dump(eval_result(unseen_gen_outputs, unseen_Gold), outfile, indent=4, ensure_ascii=False)

# Further analysis of the TCQ relation metrics, not inluded in the paper
# 1) sample 3 output in CQ generation 
all_generated_tcqs = [
    [ entry["generated_outputs"].get(str(i), "") for i in range(3) ]
    for entry in base_model_outputs
]

# 2) metric calculation
metrics = analyze_tcq_relation_metrics(all_generated_tcqs, Gold)

# 3) save the metrics
with open("Base_tcq_relation_metrics.json", "w", encoding="utf-8") as out:
    json.dump(metrics, out, ensure_ascii=False, indent=4)