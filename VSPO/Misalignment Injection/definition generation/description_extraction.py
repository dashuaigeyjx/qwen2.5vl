import os
import json
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS, OWL

# ——————————————————————————————————————————
# 1) owl files to be processed label replacement
TARGET_WITH_LABEL = [
    "swo_merged.owl",
    "OntoDT.owl"
]

# input/output directory
INPUT_DIR = "Ontology"
OUTPUT_DIR = "Ontology_description"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# oboInOwl namespace
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")


def format_node(node, g, label_map=None):
    """
    node → human-readable string
    1) if label_map[node] exists, return label_map[node]
    2) else, return qname(node)
    3) if failed, return str(node)
    """
    if label_map and node in label_map:
        return label_map[node]
    try:
        return g.qname(node)
    except Exception:
        return str(node)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".owl"):
        continue

    owl_path = os.path.join(INPUT_DIR, fname)
    g = Graph()
    g.parse(owl_path, format="xml")  # format="ttl" if using turtle format

    # mapping label
    use_label = fname in TARGET_WITH_LABEL
    label_map = {}
    if use_label:
        for subj, _, lbl in g.triples((None, RDFS.label, None)):
            label_map[subj] = str(lbl)

    # extracting class and property annotations
    class_ann_dict = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        cls_name = format_node(cls, g, label_map)
        ann = {}
        # rdfs:comment
        for comment in g.objects(cls, RDFS.comment):
            c = str(comment)
            if "rdfs:comment" not in ann:
                ann["rdfs:comment"] = c
            else:
                existing = ann["rdfs:comment"]
                if isinstance(existing, list):
                    existing.append(c)
                else:
                    ann["rdfs:comment"] = [existing, c]
        # oboInOwl:hasDefinition
        for definition in g.objects(cls, OBOINOWL.hasDefinition):
            d = str(definition)
            if "oboInOwl:hasDefinition" not in ann:
                ann["oboInOwl:hasDefinition"] = d
            else:
                existing = ann["oboInOwl:hasDefinition"]
                if isinstance(existing, list):
                    existing.append(d)
                else:
                    ann["oboInOwl:hasDefinition"] = [existing, d]
        if ann:
            class_ann_dict[cls_name] = ann

    # property annotations extraction
    prop_ann_dict = {}
    properties = list(g.subjects(RDF.type, OWL.ObjectProperty)) + list(g.subjects(RDF.type, OWL.DatatypeProperty))
    for prop in properties:
        prop_name = format_node(prop, g, label_map)
        ann = {}
        # rdfs:comment
        for comment in g.objects(prop, RDFS.comment):
            c = str(comment)
            if "rdfs:comment" not in ann:
                ann["rdfs:comment"] = c
            else:
                existing = ann["rdfs:comment"]
                if isinstance(existing, list):
                    existing.append(c)
                else:
                    ann["rdfs:comment"] = [existing, c]
        # oboInOwl:hasDefinition
        for definition in g.objects(prop, OBOINOWL.hasDefinition):
            d = str(definition)
            if "oboInOwl:hasDefinition" not in ann:
                ann["oboInOwl:hasDefinition"] = d
            else:
                existing = ann["oboInOwl:hasDefinition"]
                if isinstance(existing, list):
                    existing.append(d)
                else:
                    ann["oboInOwl:hasDefinition"] = [existing, d]
        if ann:
            prop_ann_dict[prop_name] = ann

    # save the results
    output = {
        "classes": class_ann_dict,
        "properties": prop_ann_dict
    }
    base = os.path.splitext(fname)[0]
    out_fname = f"{base}_description.json"
    out_path = os.path.join(OUTPUT_DIR, out_fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # print the result
    class_count = len(class_ann_dict)
    prop_count = len(prop_ann_dict)
    print(f"✔️ {fname} → {out_fname} saved (use_label={use_label}, #classes: {class_count}, #properties: {prop_count})")
