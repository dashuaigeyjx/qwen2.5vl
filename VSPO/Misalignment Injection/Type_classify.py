import os
import json
import re
# Define the directories
axiom_relations = [
    "subClassOf", "equivalentClass", "propertyRestrictions",
    "disjointWith", "subPropertyOf", "domain", "range",
    "characteristics", "inverseOf"
]
directories = ['Axiom_per_entity', 'Generated CQ', 'generated description']

# Load the list of files in each directory
files_in_directories = {directory: os.listdir(directory) for directory in directories}

total_data = {}
for directory in files_in_directories["Axiom_per_entity"]:
    with open(f"Axiom_per_entity/{directory}", "r") as file:
        total_data[directory.split("_")[0]] = json.load(file)

description_data = {}
for directory in files_in_directories["generated description"]:
    temp = {}
    with open(f"generated description/{directory}", "r") as file:
        for line in file:
            temp[json.loads(line)["class"]] = json.loads(line)["description"]
    description_data[directory.split("_")[0]] = temp

CQ_data = {}
for directory in files_in_directories["Generated CQ"]:
    temp = {}
    with open(f"Generated CQ/{directory}", "r") as file:
        for line in file:
            axiom = json.loads(line)["axiom"]
            for rela in axiom_relations:
                if rela in axiom:
                    cls = axiom.split(rela)[0].strip()
                    break
            if cls not in temp: temp[cls] = []
            temp[cls].append({"axiom": axiom, "CQ": json.loads(line)["CQ"]})
    CQ_data[directory.split("_")[0]] = temp


for ontology in total_data:
    for classorproperty in total_data[ontology]:
        for cp in total_data[ontology][classorproperty]:
            temp = {"axiom" : total_data[ontology][classorproperty][cp], "description" : description_data[ontology][cp], "CQ": CQ_data[ontology][cp]}
            total_data[ontology][classorproperty][cp] = temp
with open("total_data.json", "w") as json_file:
    json.dump(total_data, json_file, indent=4, ensure_ascii=False)



with open("total_data.json", "r", encoding="utf-8") as json_file:
    total_data = json.load(json_file)

# initialize types
type1, type2, type3, type4 = {}, {}, {}, {}
types = [type1, type2, type3, type4]
for t in types:
    for ontology in total_data:
        t[ontology] = {"classes": {}, "properties": {}}

class_counts = {id(t): 0 for t in types}
prop_counts  = {id(t): 0 for t in types}

def has_and_or_some_only_in_axiom(ax):
    for v in ax.values():
        if isinstance(v, list):
            for expr in v:
                if re.search(r'\b(and|or|some|only)\b', expr):
                    return True
    return False

for ontology, cps in total_data.items():
    for section in ("classes", "properties"):
        for cp, value in cps[section].items():
            ax = value.get("axiom", {}) if isinstance(value, dict) else {}

            # 1) eligible types
            if section == "classes":
                is_single_cq = (
                    "CQ" in value
                    and isinstance(value["CQ"], list)
                    and len(value["CQ"]) == 1
                )
                eligible = [type3, type4] if is_single_cq else types.copy()
            else:
                is_empty_axiom = (
                    not ax.get("characteristics")
                    and ax.get("domain") == ["None"]
                    and ax.get("range") == ["None"]
                    and not ax.get("subPropertyOf")
                    and not ax.get("inverseOf")
                )
                eligible = [type3, type4] if is_empty_axiom else types.copy()


            #  2) axiom including and/or some/only processing 
            if not has_and_or_some_only_in_axiom(ax):
                eligible = [t for t in eligible if t is not type3]

            # 3) classification
            if section == "classes":
                target = min(eligible, key=lambda t: class_counts[id(t)])
                class_counts[id(target)] += 1
            else:
                target = min(eligible, key=lambda t: prop_counts[id(t)])
                prop_counts[id(target)] += 1

            target[ontology][section][cp] = value

# results
for idx, t in enumerate(types, start=1):
    nc = sum(len(t[ont]["classes"]) for ont in t)
    np = sum(len(t[ont]["properties"]) for ont in t)
    print(f"type{idx} ➔ {nc} classes, {np} properties")

# save to json files
with open("type1.json", "w", encoding="utf-8") as f:
    json.dump(type1, f, ensure_ascii=False, indent=2)
with open("type2.json", "w", encoding="utf-8") as f:
    json.dump(type2, f, ensure_ascii=False, indent=2)
with open("type3.json", "w", encoding="utf-8") as f:
    json.dump(type3, f, ensure_ascii=False, indent=2)
with open("type4.json", "w", encoding="utf-8") as f:
    json.dump(type4, f, ensure_ascii=False, indent=2)
