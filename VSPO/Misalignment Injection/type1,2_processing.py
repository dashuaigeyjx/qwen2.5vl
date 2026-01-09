import json
import random

# List of possible axiom predicates
axiom_relations = [
    "subClassOf", "equivalentClass", "propertyRestrictions",
    "disjointWith", "subPropertyOf", "domain", "range",
    "characteristics", "inverseOf"
]

def process_type1(input_path='type1.json', output_path='processed_type1.json'):
    # 1. Load the original data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Traverse each ontology
    for ontology in data.values():
        # process both classes and properties
        for section in ('classes', 'properties'):
            for name, info in ontology.get(section, {}).items():
                all_cq = info.get('CQ', [])
                filtered = []
                for entry in all_cq:
                    ax = entry['axiom']
                    # if the axiom is empty or contains "None" in domain/range, skip it
                    if " domain None" in ax or " range None" in ax:
                        continue
                    filtered.append(entry)
                # if no CQ entries are found, skip this entity
                cq_entries = filtered if filtered else all_cq
                if not cq_entries:
                    continue
                
                if section == 'classes':
                    preferred = [e for e in cq_entries if ' disjointWith ' in e['axiom']]
                else:  # section == 'properties'
                    preferred = [e for e in cq_entries if ' inverseOf ' in e['axiom']]
                
                if preferred:
                    sampled = random.choice(preferred)
                else:
                    sampled = random.choice(cq_entries)

                # b) Parse the sampled axiom to find predicate and expression
                axiom_str = sampled['axiom']
                info['removed axiom'] = axiom_str
                predicate = None
                for rel in axiom_relations:
                    if f" {rel} " in axiom_str:
                        predicate = rel
                        break
                if predicate is None:
                    # couldn't parse, skip removal
                    continue

                # split into subject and expression
                subject, expr = axiom_str.split(f" {predicate} ", 1)
                subject = subject.strip()
                expr = expr.strip()

                # c) Remove that expression from the matching list in info['axiom'],
                #    but skip deleting "None" from domain/range
                ax = info.get('axiom', {})
                if predicate in ax and isinstance(ax[predicate], list):
                    # if it's domain/range and expr is the literal "None", skip removal
                    if predicate in ('domain', 'range') and expr == "None":
                        # do not remove the 'None' placeholder
                        pass
                    else:
                        try:
                            ax[predicate].remove(expr)
                        except ValueError:
                            print(f"Expression '{expr}' not found in axiom list for {name}.")
                            # ignore if not present

                # d) Build Target CQ and Valid CQ without deleting the original 'CQ'
                target_cq = sampled['CQ']
                valid_cq = [
                    question
                    for entry in all_cq
                    if entry is not sampled
                    for question in entry['CQ']
                ]

                # e) Attach new fields
                info['Target CQ'] = target_cq
                info['Valid CQ'] = valid_cq

    # 3. Write out the processed file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# run for both type1 and type2
process_type1(input_path='type classification/type1.json', output_path='Final_type1.json')
process_type1(input_path='type classification/type2.json', output_path='processed_type2.json')
