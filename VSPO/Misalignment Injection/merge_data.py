import json, os
import random
import copy
total_dataset = {}
train_dataset = {}
test_dataset = {}
# Load JSON files from the "Final types" directory
final_types_directory = "processed types"
for file_name in os.listdir(final_types_directory):
    if file_name.endswith(".json"):  
        with open(os.path.join(final_types_directory, file_name), "r", encoding="utf-8") as json_file:
            temp_dataset = json.load(json_file)
            for ontology in temp_dataset:
                if ontology not in total_dataset:
                    total_dataset[ontology] = {}
                for classorprop in temp_dataset[ontology]:
                    if classorprop not in total_dataset[ontology]:
                        total_dataset[ontology][classorprop] = {}
                    for cp in temp_dataset[ontology][classorprop]:
                        if cp not in total_dataset[ontology][classorprop]:
                            total_dataset[ontology][classorprop][cp] = {}
                        # Merge the data
                        total_dataset[ontology][classorprop][cp].update(temp_dataset[ontology][classorprop][cp])
                        # Split the data into train and test datasets (9:1 ratio) for each file name
                    items = list(temp_dataset[ontology][classorprop].items())
                    random.shuffle(items)
                    split_index = int(len(items) * 0.9)
                    train_items = dict(items[:split_index])
                    test_items = dict(items[split_index:])

                    if ontology not in train_dataset:
                        train_dataset[ontology] = {}
                    if ontology not in test_dataset:
                        test_dataset[ontology] = {}

                    if classorprop not in train_dataset[ontology]:
                        train_dataset[ontology][classorprop] = {}
                    if classorprop not in test_dataset[ontology]:
                        test_dataset[ontology][classorprop] = {}

                    for cp in train_items: train_dataset[ontology][classorprop][cp] = train_items[cp]
                    for cp in test_items: test_dataset[ontology][classorprop][cp] = test_items[cp]
# Save the merged data to a new JSON file
with open("merged dataset/Final_dataset.json", "w", encoding="utf-8") as json_file:
    json.dump(total_dataset, json_file, indent=4, ensure_ascii=False)
with open("merged dataset/Final_train_dataset.json", "w", encoding="utf-8") as json_file:
    json.dump(train_dataset, json_file, indent=4, ensure_ascii=False)
with open("merged dataset/Final_test_dataset.json", "w", encoding="utf-8") as json_file:
    json.dump(test_dataset, json_file, indent=4, ensure_ascii=False)

def dataset_construct(ontology, type, class_name,description, axiom, TCQ, VCQ, Taxiom, datatype, CQ):
    return {"data":{"input": f"""As an ontology engineer, generate a list of competency questions based on the following description and axiom.
Definition of competency questions: the questions that outline the scope of ontology and provide an idea about the knowledge that needs to be entailed in the ontology.
Avoid using narrative questions + axioms.
Don't generate unnecessary text. Output only the questions, separated by ` | ` (pipe with spaces). 
{type} name: {class_name}
Description: {description}
Axiom: {axiom}
Generated CQs:""",
            "output": f"{TCQ[0]} | {TCQ[1]} | {TCQ[2]} "},
            "metadata": {
            "ontology": ontology,
            "class": class_name,
            "axiom": axiom,
            "datatype": datatype,
            "TCQ": TCQ,
            "VCQ": VCQ,
            "Taxiom": Taxiom,
            "CQ" : CQ
                }}
def save_dataset(dataset, output_path):
    for ontology in dataset:
        for classorprop in dataset[ontology]:
            for cp in dataset[ontology][classorprop]:
                if classorprop == "classes":
                    type = "Class"
                else:
                    type = "Property"
                axiom = dataset[ontology][classorprop][cp]["axiom"]
                description = dataset[ontology][classorprop][cp]["description"]
                TCQ = dataset[ontology][classorprop][cp]["Target CQ"]
                VCQ = dataset[ontology][classorprop][cp]["Valid CQ"]
                Taxiom = dataset[ontology][classorprop][cp]["removed axiom"] if "removed axiom" in dataset[ontology][classorprop][cp] else "None"
                datatype = dataset[ontology][classorprop][cp]["type"]
                CQ = dataset[ontology][classorprop][cp]["CQ"]
                for cq in TCQ:
                    if cq not in VCQ:
                        VCQ.append(cq)
                line_data = dataset_construct(ontology, type, cp,description,axiom, TCQ, VCQ, Taxiom, datatype, CQ)["data"]
                line_metadata = dataset_construct(ontology, type, cp,description,axiom, TCQ, VCQ, Taxiom, datatype, CQ)["metadata"]
                with open(f"{output_path}.jsonl", "a", encoding="utf-8") as jsonl_file:
                    jsonl_file.write(json.dumps(line_data, ensure_ascii=False) + "\n")
                with open(f"{output_path}_meta.jsonl", "a", encoding="utf-8") as jsonl_file:
                    jsonl_file.write(json.dumps(line_metadata, ensure_ascii=False) + "\n")
save_dataset(train_dataset, "train_dataset")
save_dataset(test_dataset, "test_dataset")

#Generalizablility setting(unseen ontology)
onto_list = {"AWO": ["AfricanWildlifeOntology1"],
"OntoDT": ["OntoDT"],"SWO": ["swo"],"Pizza": ["pizza"],"Stuff": ["stuff"],
"DEM@Care": ["lab", "time", "home", "exchangemodel", "event"]}
with open("merged dataset/Final_dataset.json", "r", encoding="utf-8") as json_file:
    total_data = json.load(json_file)

for onto in onto_list:
    temp_train = copy.deepcopy(total_data)
    temp_test = {}
    for owl in onto_list[onto]:
        temp_test[owl] = copy.deepcopy(total_data[owl])
        del temp_train[owl]
    save_dataset(temp_train, f"additional settings/Generalizability/unseen ontology/{onto}/train_dataset")
    save_dataset(temp_test, f"additional settings/Generalizability/unseen ontology/{onto}/test_dataset")