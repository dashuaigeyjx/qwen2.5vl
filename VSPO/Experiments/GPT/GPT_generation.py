import json
from openai import OpenAI
def init_template(input):
    return {
    "custom_id": None, 
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4.1", 
             "messages": 
                 [
                   {"role": "user", "content": f"{input}"},
                  ],
             "max_tokens": 120 
             }
    }
# Load data from Final_test_dataset.jsonl
with open("test_dataset.jsonl", "r") as file:
    test_dataset = [json.loads(line) for line in file]

with open("test_dataset_meta.jsonl", "r") as file:
    test_meta = [json.loads(line) for line in file]
batches = []
for i, data in enumerate(test_dataset):
    input_text = data["input"]
    batch = init_template(input_text)
    batch.update({"custom_id": f"{test_meta[i]['ontology']}_{test_meta[i]['class']}"})
    batches.append(batch)
with open("GPT_input.jsonl", 'w', encoding='utf-8') as file:
        for item in batches:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')

client = OpenAI(api_key="") # Initialize the OpenAI client

with open("GPT_input.jsonl", "rb") as file:
  batch_input_file = client.files.create(
    file=file, 
    purpose="batch"
  )
batch_input_file_id = batch_input_file.id

# Create a batch job
client.batches.create(
  input_file_id=batch_input_file_id,
  endpoint="/v1/chat/completions",
  completion_window="24h", 
        )
