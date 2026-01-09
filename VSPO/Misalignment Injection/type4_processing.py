import json
import random
def process_type4(input_path='type classification/type4.json', output_path='Final_type4.json'):
    # 1. Load the original data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    over2_axiom_num =0
    # 3. Process each ontology
    for ontology in data.values():
        for section in ('classes', 'properties'):
            for name, info in ontology.get(section, {}).items():
                temp_list = []
                for entry in info.get('CQ', []):
                    for cq in entry['CQ']:
                        if cq not in temp_list:
                            temp_list.append(cq)

                info['Target CQ'] = random.sample(temp_list, min(3, len(temp_list)))
                info['Valid CQ'] = temp_list
                if len(temp_list)>3: over2_axiom_num+=1
    print("over2_axiom_num: ", over2_axiom_num)
                    

    # 4. Write out the processed file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
process_type4()