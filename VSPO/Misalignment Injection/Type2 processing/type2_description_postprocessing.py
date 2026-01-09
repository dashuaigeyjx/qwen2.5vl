import os
import json

jsonl_files = ["Batchoutput.jsonl"]

for jsonl_file in jsonl_files:
    loaded_data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            temp_data = json.loads(line)
            temp_out = {"class": temp_data["custom_id"].split("_",1)[1], "description": temp_data["response"]["body"]["choices"][0]["message"]["content"]}
            loaded_data.append(temp_out)
    # Save the loaded data to a new JSONL file
    with open("Generated_description.jsonl", "w", encoding="utf-8") as f:
        for item in loaded_data:
            json_string = json.dumps(item, ensure_ascii=False)
            f.write(json_string + '\n')

with open("processed_type2.json", "r", encoding="utf-8") as outfile:
    total_type2 = json.load(outfile)
with open("Generated_description.jsonl", "r", encoding="utf-8") as f:
    temp_desc = {}
    for line in f:
        loadfile = json.loads(line)
        temp_desc[loadfile["class"]] = loadfile["description"]
for ontology in total_type2:
    for cls in total_type2[ontology]["classes"]:
        total_type2[ontology]["classes"][cls]["description"] = temp_desc[cls]
    for prop in total_type2[ontology]["properties"]:
        total_type2[ontology]["properties"][prop]["description"] = temp_desc[prop]
with open("type2_description_update.json", "w", encoding="utf-8") as outfile:
    json.dump(total_type2, outfile, ensure_ascii=False, indent=4)



# List of axiom predicates to look for
axiom_relations = [
    "subClassOf", "equivalentClass", "propertyRestrictions",
    "disjointWith", "subPropertyOf", "domain", "range",
    "characteristics", "inverseOf"
]

def restore_removed_axioms(input_path='type2_description_update.json', output_path='Final_type2.json'):
    # 1. Load the processed data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Traverse each ontology, classes and properties
    for ontology in data.values():
        for section in ('classes', 'properties'):
            for name, info in ontology.get(section, {}).items():
                removed = info.get('removed axiom')
                if not removed:
                    continue

                # find predicate
                predicate = next((rel for rel in axiom_relations if f" {rel} " in removed), None)
                if predicate is None:
                    continue

                # split into subject and expression
                _, expr = removed.split(f" {predicate} ", 1)
                expr = expr.strip()

                # restore into info['axiom'][predicate]
                ax = info.setdefault('axiom', {})
                current = ax.setdefault(predicate, [])
                if isinstance(current, list) and expr not in current:
                    current.append(expr)

    # 3. Write out the restored file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Example usage:
restore_removed_axioms(
    input_path='type2_description_update.json',
    output_path='Final_type2.json'
)
