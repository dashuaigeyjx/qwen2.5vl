import os
import json
from copy import deepcopy
from openai import OpenAI

def load_templates(directory="templates"):
    """
    Read all .txt files in the given directory and return a dict
    mapping filename (without .txt) to file content.
    """
    templates = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(".txt"):
            key = os.path.splitext(filename)[0]
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                templates[key] = f.read()
    return templates
def init_template(axiom, axiom_relation):
    if axiom_relation in ["domain", "range"]: template = load_templates()["domain_range"]
    else: template = load_templates()[axiom_relation]
    # Add a hint for the logic of the axiom
    logic_hint = ""
    if any(keyword in axiom for keyword in ["propertyRestrictions", "equivalentClass"]):
        logic_hint = """
        The axiom may include different logical structures. Determine whether it involves existential/universal restrictions (some/only) or intersection/union (and/or), and generate CQs accordingly.
        """
    elif any(keyword in axiom for keyword in ["domain", "range"]):
        logic_hint = """
        If the property’s domain or range is undefined (None), generate a Competency Question asking what can be the domain or range of the property, is it right that the property has no domain or range.
        """
    return {
        "custom_id": None,  
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1",  
            "messages": [
                {
                    "role": "system",
                    "content": f"""
                    As an ontology engineer, generate a list of competency questions based on the following axiom and one-shot example.
                    Definition of competency questions: the questions that outline the scope of ontology and provide an idea about the knowledge that needs to be entailed in the ontology.
                    Avoid using narrative questions + axioms.
                    Don't generate unnecessary text. Just return 3 distinct CQs separated by ' // '.
                    Use the one-shot and known templates only as inspiration — do not copy them directly. Rephrase and vary the structure of each CQ while maintaining its logical intent.
                    {logic_hint.strip()}
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Generate competency questions including axioms and current template.
                    Template: {template}
                    Axiom: {axiom}
                    """
                },
                {
                    "role": "user",
                    "content": "Generated CQs:"
                }
            ],
            "max_tokens": 512
        }
    }


directory_path = "Axiom_per_entity"
ontology_list = {file.split("_")[0]: os.path.join(directory_path, file) for file in os.listdir(directory_path)}


data_num = 0
for ontology in ontology_list:
    batches = []
    seen_ids = set()
    with open(ontology_list[ontology], "r") as f:
        temp_axiom=json.load(f)
    for cls in temp_axiom["classes"]:
        for axiom_relation in temp_axiom["classes"][cls]:
            for axiom_range in temp_axiom["classes"][cls][axiom_relation]:
                axiom = f"{cls} {axiom_relation} {axiom_range}"
                custom_id = f"{ontology}_{axiom}"
                if custom_id in seen_ids:
                    continue
                seen_ids.add(custom_id)
                temp = deepcopy(init_template(axiom=axiom, axiom_relation=axiom_relation))
                temp["custom_id"] = custom_id
                batches.append(temp)
    for prop in temp_axiom["properties"]:
        for axiom_relation in temp_axiom["properties"][prop]:
            for axiom_range in temp_axiom["properties"][prop][axiom_relation]:
                axiom = f"{prop} {axiom_relation} {axiom_range}"
                custom_id = f"{ontology}_{axiom}"
                if custom_id in seen_ids:
                    continue
                seen_ids.add(custom_id)
                temp = deepcopy(init_template(axiom=axiom, axiom_relation=axiom_relation))
                temp["custom_id"] = custom_id
                batches.append(temp)
    # Save the batches to a JSONL file
    with open("CQ_Batchinput/"+ontology+'_batchinput.jsonl', 'w', encoding='utf-8') as file:
        for item in batches:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')
        print(f"Ontology name: {ontology}, Number of batches: {len(batches)}")
    data_num += len(batches)
print(f"Total number of batches: {data_num}")

client = OpenAI(api_key="")

for ontology in os.listdir("CQ_Batchinput"):
  if ontology.endswith(".jsonl"):
    with open(os.path.join("CQ_Batchinput", ontology), "rb") as file:
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
