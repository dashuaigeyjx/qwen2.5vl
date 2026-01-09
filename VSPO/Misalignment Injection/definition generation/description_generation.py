import os
import json
from copy import deepcopy
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
                   You are an Ontology engineer. 
                   Generate a {type} description including information of axioms and current description. 
                   The description should be concise and informative, providing a clear understanding of the {type}'s purpose and characteristics.
                   Don't generate unnecessary text. Just generate {type} description only."""},
                   {"role": "user", "content": f"""
                   Generate a {type} description including information of axioms and current description. 
                   If there is no description, generate a description based on the axiom."""},
                   {"role": "user", "content": "For example, "+example},
                   {"role": "user", "content": "Now, generate the description using class axiom and current description below."}
                  ],
             "max_tokens": 1024 
             }
    }

directory_path = "Ontology_description"
ontology_list = {file.split("_")[0]: {"description":os.path.join(directory_path, file)} for file in os.listdir(directory_path)}
for file in os.listdir("Axiom_per_entity"):
    ontology_list[file.split("_")[0]]["axiom"]=os.path.join("Axiom_per_entity", file)

data_num = 0
for ontology in ontology_list:
    batches = []
    cls_num, pro_num = 0, 0
    with open(ontology_list[ontology]["description"], "r") as f:
        temp_desc=json.load(f)
    with open(ontology_list[ontology]["axiom"], "r") as f:
        temp_axiom=json.load(f)
    for cls in temp_axiom["classes"]:
        temp = deepcopy(init_template("class", C_example))
        temp["body"]["messages"].append({"role": "user", "content": "Class name: "+cls})
        temp["custom_id"] = ontology+"_"+cls
        temp["body"]["messages"].append({"role": "user", "content": "Axiom: "+str(temp_axiom["classes"][cls])})
        if cls in temp_desc["classes"]: temp["body"]["messages"].append({"role": "user", "content": "Current description: "+str(temp_desc["classes"][cls])})
        temp["body"]["messages"].append({"role": "user", "content": "Generated description: "})
        batches.append(temp)
    cls_num=len(batches)
    print(f"Ontology name: {ontology}, {cls_num} classes")
    for prop in temp_axiom["properties"]:
        temp = deepcopy(init_template("property", P_example))
        temp["body"]["messages"].append({"role": "user", "content": "Property name: "+prop})
        temp["custom_id"] = ontology+"_"+prop
        temp["body"]["messages"].append({"role": "user", "content": "Axiom: "+str(temp_axiom["properties"][prop])})
        if prop in temp_desc["properties"]: temp["body"]["messages"].append({"role": "user", "content": "Current description: "+str(temp_desc["properties"][prop])})
        temp["body"]["messages"].append({"role": "user", "content": "Generated description: "})
        pro_num+=1
        batches.append(temp)
    
    pro_num=len(batches)-cls_num
    print(f"Ontology name: {ontology}, {pro_num} properties")
    # Save the batches to a JSONL file
    with open("Batchinput/"+ontology+'_batchinput.jsonl', 'w', encoding='utf-8') as file:
        for item in batches:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')
    print(f"Ontology name: {ontology}, Number of batches: {len(batches)}")
    data_num += len(batches)
print(f"Total number of batches: {data_num}")
    

client = OpenAI(api_key="")

for ontology in os.listdir("Batchinput"):
  if ontology.endswith("stuff_batchinput.jsonl"):
    with open(os.path.join("Batchinput", ontology), "rb") as file:
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
  else:
    print(f"Skipping non-JSONL file: {ontology}")