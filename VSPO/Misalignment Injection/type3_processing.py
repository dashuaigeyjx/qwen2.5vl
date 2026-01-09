import re
import random
import json

def _mutate_axiom(expr: str,
                  swap_some_only=True,
                  swap_and_or=True) -> str:
    
    stack = ['']
    brackets = []
    for ch in expr:
        if ch in '([{':
            stack.append('')
            brackets.append(ch)
        elif ch in ')]}':
            inner = stack.pop()
            open_br = brackets.pop()
            mutated_inner = _mutate_axiom(inner, swap_some_only, swap_and_or)
            close_br = {'(': ')', '[': ']', '{': '}'}[open_br]
            stack[-1] += f'{open_br}{mutated_inner}{close_br}'
        else:
            stack[-1] += ch

    level_str = stack[0]

    if swap_and_or and re.search(r'\band\b|\bor\b', level_str):
        if random.random() < 0.5:
            level_str = re.sub(r'\band\b', '__TMP_AND__', level_str)
            level_str = re.sub(r'\bor\b', 'and', level_str)
            level_str = re.sub(r'__TMP_AND__', 'or', level_str)

    if swap_some_only and random.random() < 0.5:
        matches = list(re.finditer(r'\bsome\b|\bonly\b', level_str))
        if matches:
            m = random.choice(matches)
            orig = m.group(0)
            repl = 'only' if orig == 'some' else 'some'
            level_str = level_str[:m.start()] + repl + level_str[m.end():]

    return level_str

def mutate_axiom(expr: str,
                 swap_some_only=True,
                 swap_and_or=True) -> str:
    # This function mutates an axiom expression by swapping "some" with "only" 
    # and/or swapping "and" with "or" based on the provided flags.
    mutated = _mutate_axiom(expr, swap_some_only, swap_and_or)
    if mutated == expr:
        if re.search(r'\band\b|\bor\b', expr):
            mutated = re.sub(r'\band\b', '__TMP_AND__', expr)
            mutated = re.sub(r'\bor\b', 'and', mutated)
            mutated = re.sub(r'__TMP_AND__', 'or', mutated)
        elif re.search(r'\bsome\b|\bonly\b', expr):
            matches = list(re.finditer(r'\bsome\b|\bonly\b', expr))
            m = random.choice(matches)
            orig = m.group(0)
            repl = 'only' if orig == 'some' else 'some'
            mutated = expr[:m.start()] + repl + expr[m.end():]
    return mutated


# List of possible axiom predicates
axiom_relations = [
    "subClassOf", "equivalentClass", "propertyRestrictions",
    "disjointWith", "subPropertyOf", "domain", "range",
    "characteristics", "inverseOf"
]

def process_type3(input_path='type classification/type3.json',
                  output_path='Final_type3.json'):
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
                    if " domain None" in ax and " range None" in ax:
                        continue
                    filtered.append(entry)
                cq_entries = filtered if filtered else all_cq
                if not cq_entries:
                    continue

                # a) extract CQ entries with "and" or "or" or "some" or "only"
                candidates = [
                    entry for entry in cq_entries
                    if re.search(r'\b(and|or|some|only)\b', entry['axiom'])
                ]
                if not candidates:
                    # if and/or/some/only are not present in any axiom, skip this entity
                    continue
                sampled = random.choice(candidates)

                # b) Parse the sampled axiom to find predicate and expression
                axiom_str = sampled['axiom']
                # mutate the axiom
                info['editted axiom'] = mutate_axiom(axiom_str)
                info['removed axiom'] = axiom_str

                predicate = None
                for rel in axiom_relations:
                    if f" {rel} " in axiom_str:
                        predicate = rel
                        break
                if predicate is None:
                    # couldn't parse, skip
                    continue

                # split into subject and original expression
                subject, expr = axiom_str.split(f" {predicate} ", 1)
                expr = expr.strip()

                # split into edited expression
                _, editted_expr = info['editted axiom'].split(f" {predicate} ", 1)
                editted_expr = editted_expr.strip()

                # c) change the original expression to the edited expression 
                ax = info.get('axiom', {})
                if predicate in ax and isinstance(ax[predicate], list):
                    # if it's domain/range and expr is the literal "None", skip removal
                    if not (predicate in ('domain', 'range') and expr == "None"):
                        try:
                            ax[predicate].remove(expr)
                            ax[predicate].append(editted_expr)
                        except ValueError:
                            print(f"Expression '{expr}' not found in axiom list for {name}.")

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

process_type3()
