from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Any


def evaluate_all_cq_generations(
    all_generated_cqs: list[list[str]],
    all_target_cqs: list[list[str]],
    model_name: str = 'all-MiniLM-L6-v2',
    threshold: float = 0.7,
    use_cuda: bool = True
) -> dict:
    

    assert len(all_generated_cqs) == len(all_target_cqs), "different data sumber."

    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    # flatten
    all_gen_flat, all_tgt_flat = [], []
    gen_group_ids, tgt_group_ids = [], []

    for i, (gen_cqs, tgt_cqs) in enumerate(zip(all_generated_cqs, all_target_cqs)):
        all_gen_flat.extend(gen_cqs)
        all_tgt_flat.extend(tgt_cqs)
        gen_group_ids.extend([i] * len(gen_cqs))
        tgt_group_ids.extend([i] * len(tgt_cqs))

    emb_gen = model.encode(all_gen_flat, convert_to_tensor=True, device=device)
    emb_tgt = model.encode(all_tgt_flat, convert_to_tensor=True, device=device)

    cos_scores = util.cos_sim(emb_gen, emb_tgt)

    n_gen_total = len(all_gen_flat)
    n_tgt_total = len(all_tgt_flat)

    matched_gen = torch.zeros(n_gen_total, dtype=torch.bool)
    matched_tgt = torch.zeros(n_tgt_total, dtype=torch.bool)

    max_scores_per_gen = [0.0] * n_gen_total

    # generated CQ matching
    for i in range(n_gen_total):
        valid_tgt = [j for j, gid in enumerate(tgt_group_ids) if gid == gen_group_ids[i]]
        if valid_tgt:
            scores = cos_scores[i][valid_tgt]
            max_score = float(torch.max(scores))
            max_scores_per_gen[i] = max_score
            if max_score >= threshold:
                matched_gen[i] = True

    # semantic pitfall CQ matching
    for j in range(n_tgt_total):
        valid_gen = [i for i, gid in enumerate(gen_group_ids) if gid == tgt_group_ids[j]]
        if valid_gen:
            if float(torch.max(cos_scores[valid_gen, j])) >= threshold:
                matched_tgt[j] = True

    # metric calculation
    tp_gen = matched_gen.sum().item()
    tp_tgt = matched_tgt.sum().item()

    precision = tp_gen / n_gen_total if n_gen_total > 0 else 0.0
    recall = tp_tgt / n_tgt_total if n_tgt_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_max_cosine = sum(max_scores_per_gen) / n_gen_total if n_gen_total > 0 else 0.0

    # transform to percentage
    precision = round(precision * 100, 1)
    recall = round(recall * 100, 1)
    f1 = round(f1 * 100, 1)
    avg_max_cosine = round(avg_max_cosine, 4)

    return {
        'precision (%)':         precision,
        'recall (%)':            recall,
        'f1_score (%)':          f1,
        'avg_max_cosine_per_gen': avg_max_cosine
    }

def eval_result(generated_output, gold):
    gen_type = {"Type1": [], "Type2": [], "Type3": [], "Type4": []}
    VCQ_type = {"Type1": [], "Type2": [], "Type3": [], "Type4": []}
    GCQ_type = {"Type1": [], "Type2": [], "Type3": []}
    for n, data in enumerate(gold):
        if data["datatype"] == "Type4":
            gen_type[data["datatype"]].append([generated_output[n]["generated_outputs"]])
            VCQ_type[data["datatype"]].append(data["VCQ"])
        else:
            gen_type[data["datatype"]].append([generated_output[n]["generated_outputs"]])
            VCQ_type[data["datatype"]].append(data["VCQ"])
            GCQ_type[data["datatype"]].append(data["TCQ"])
    gen_VCQ = gen_type["Type1"] + gen_type["Type2"] + gen_type["Type3"] + gen_type["Type4"]
    gen_GCQ = gen_type["Type1"] + gen_type["Type2"] + gen_type["Type3"]
    VCQ = VCQ_type["Type1"] + VCQ_type["Type2"] + VCQ_type["Type3"] + VCQ_type["Type4"]
    GCQ = GCQ_type["Type1"] + GCQ_type["Type2"] + GCQ_type["Type3"]
    result = {}
    result["VCQ"] = evaluate_all_cq_generations(gen_VCQ, VCQ)
    result["GCQ"] = evaluate_all_cq_generations(gen_GCQ, GCQ)
    for key in gen_type.keys():
        if key == "Type4":
            result[key+"_VCQ"] = evaluate_all_cq_generations(gen_type[key], VCQ_type[key])
        else:
            result[key+"_VCQ"] = evaluate_all_cq_generations(gen_type[key], VCQ_type[key])
            result[key+"_GCQ"] = evaluate_all_cq_generations(gen_type[key], GCQ_type[key])
    return result
    




# axiom relations
AXIOM_RELATIONS = [
    "subClassOf", "equivalentClass", "propertyRestrictions",
    "disjointWith", "subPropertyOf", "domain", "range",
    "characteristics", "inverseOf"
]

def match_generated_cqs_to_axioms(
    gen_cqs: List[str],
    cq_entries: List[Dict[str, Any]],
    model: SentenceTransformer,
    device: str
) -> List[Dict[str, Any]]:
    
    emb_gen = model.encode(gen_cqs, convert_to_tensor=True, device=device)
    axiom_embs = []
    for ax in cq_entries:
        cqs = ax["CQ"]
        if not cqs: 
            continue
        emb_tgt = model.encode(cqs, convert_to_tensor=True, device=device)
        axiom_embs.append((ax["axiom"], emb_tgt))
    matches = []
    for i, gen in enumerate(gen_cqs):
        best_score = -1.0
        best_axiom = None
        for ax_str, emb_tgt in axiom_embs:
            score = float(torch.max(util.cos_sim(emb_gen[i].unsqueeze(0), emb_tgt)))
            if score > best_score:
                best_score = score
                best_axiom = ax_str
        matches.append({
            "generated":     gen,
            "matched_axiom": best_axiom,
            "score":         best_score
        })
    return matches

def analyze_tcq_relation_metrics(
    all_generated_cqs: List[List[str]],
    gold_data: List[Dict[str, Any]],
    model_name: str = 'all-MiniLM-L6-v2',
    use_cuda: bool = True
) -> Dict[str, Dict[str, Dict[str, float]]]:
    
    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    model  = SentenceTransformer(model_name, device=device)

    # initialize counters
    types = ["Type1", "Type2", "Type3"]
    pred_counts = {t: {r: 0 for r in AXIOM_RELATIONS} for t in types}
    true_counts = {t: {r: 0 for r in AXIOM_RELATIONS} for t in types}
    tp_counts   = {t: {r: 0 for r in AXIOM_RELATIONS} for t in types}

    # iterate over each entry
    for gen_cqs, entry in zip(all_generated_cqs, gold_data):
        dtype = entry.get("datatype")
        if dtype not in types:
            continue

        # ground-truth axiom for TCQ
        gt_axiom = entry.get("Taxiom", "")
        true_rel = next((r for r in AXIOM_RELATIONS if r in gt_axiom), None)
        if true_rel is None:
            continue

        # match each generated CQ to its best axiom
        matches = match_generated_cqs_to_axioms(gen_cqs, entry.get("CQ", []), model, device)

        # for each generated CQ, update counts
        for m in matches:
            pred_axiom = m["matched_axiom"] or ""
            pred_rel   = next((r for r in AXIOM_RELATIONS if r in pred_axiom), None)
            if pred_rel is None:
                continue

            pred_counts[dtype][pred_rel]    += 1
            true_counts[dtype][true_rel]    += 1
            if pred_rel == true_rel:
                tp_counts[dtype][true_rel]  += 1

    # compute micro precision/recall/f1 per relation per type
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for t in types:
        results[t] = {}
        for rel in AXIOM_RELATIONS:
            tp = tp_counts[t][rel]
            pc = pred_counts[t][rel]
            tc = true_counts[t][rel]
            if tc == 0:
                # if this type has no relation, exclude from statistics
                continue
            precision = (tp / pc) if pc > 0 else 0.0
            recall    = tp / tc
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            # transform to percentage
            results[t][rel] = {
                "precision (%)": round(precision * 100, 2),
                "recall (%)":    round(recall * 100, 2),
                "f1_score (%)":  round(f1 * 100, 2)
            }

    return results