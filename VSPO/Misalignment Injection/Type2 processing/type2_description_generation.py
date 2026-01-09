import json
from copy import deepcopy
import os
from openai import OpenAI
with open("C_example.txt", "r") as file:
	C_example = file.read()
with open("P_example.txt", "r") as file:
  P_example = file.read()
def init_template(type, example):
    return {
    "custom_id": None, 
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4.1", 
             "messages": 
                 [{"role": "system", "content": f"""
                   You are a helpful Ontology engineer. 
                   Generate a {type} description including information of axioms. 
                   The description should be concise and informative, providing a clear understanding of the {type}'s purpose and characteristics.
                   Don't generate unnecessary text. Just generate {type} description only."""},
                   {"role": "user", "content": f"""
                   Generate a {type} description including information of axioms. 
                   If there is no description, generate a description based on the axiom."""},
                   {"role": "user", "content": "For example, "+example},
                   {"role": "user", "content": "Now, generate the description using class axiom below."}
                  ],
             "max_tokens": 1024 
             }
    }
directory_path = "Ontology_description"
ontology_list = {file.split("_")[0]: {"description":os.path.join(directory_path, file)} for file in os.listdir(directory_path)}


data_num = 0
batches = []
for ontology in ontology_list:
    with open("processed_type2.json", "r") as f:
        temp_axiom=json.load(f)[ontology]
    for cls in temp_axiom["classes"]:
        temp = deepcopy(init_template("class", C_example))
        temp["body"]["messages"].append({"role": "user", "content": "Class name: "+cls})
        temp["custom_id"] = ontology+"_"+cls
        temp["body"]["messages"].append({"role": "user", "content": "Axiom: "+str(temp_axiom["classes"][cls]["axiom"])})
        temp["body"]["messages"].append({"role": "user", "content": "Generated description: "})
        batches.append(temp)
    for prop in temp_axiom["properties"]:
        temp = deepcopy(init_template("property", P_example))
        temp["body"]["messages"].append({"role": "user", "content": "Property name: "+prop})
        temp["custom_id"] = ontology+"_"+prop
        temp["body"]["messages"].append({"role": "user", "content": "Axiom: "+str(temp_axiom["properties"][prop]["axiom"])})
        temp["body"]["messages"].append({"role": "user", "content": "Generated description: "})
        batches.append(temp)
    data_num += len(batches)
# Save the batches to a JSONL file
with open("Batchinput.jsonl", 'w', encoding='utf-8') as file:
    for item in batches:
        json_string = json.dumps(item, ensure_ascii=False)
        file.write(json_string + '\n')
    
    

client = OpenAI(api_key="")


with open("Batchinput.jsonl", "rb") as file:
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
