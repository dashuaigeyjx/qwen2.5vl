import sys
import os
import json
import rdflib
from rdflib import Graph, RDF, RDFS, OWL
from rdflib.namespace import XSD
import random
from rdflib import BNode, URIRef
# ——————————————
# owl files to be processed label replacement
TARGET_WITH_LABEL = [
    "swo_merged.owl",
    "OntoDT.owl"
]

# output directory
PREFIX_DIR = "Prefixes"
AXIOM_DIR = "Axiom_per_entity"
INPUT_DIR = "Ontology"
for d in (PREFIX_DIR, AXIOM_DIR):
    os.makedirs(d, exist_ok=True)


def format_node(node, g, label_map=None):
    if label_map and node in label_map:
        return label_map[node]
    try:
        return g.qname(node)
    except Exception:
        return str(node)


def parse_rdf_list(list_node, g):
    items = []
    while list_node and list_node != RDF.nil:
        first = g.value(list_node, RDF.first)
        if first is None:
            break
        items.append(first)
        list_node = g.value(list_node, RDF.rest)
        if list_node is None:
            break
    return items


def process_class_expression(node, g, label_map=None):
    """
    Processes an OWL class expression node and converts it into a human-readable string representation.

    Args:
        node (rdflib.term.Node): The RDF node representing the OWL class expression.
        g (rdflib.Graph): The RDF graph containing the ontology data.
        label_map (dict, optional): A mapping of RDF nodes to human-readable labels. Defaults to None.

    Returns:
        str: A string representation of the OWL class expression.

    The function handles the following OWL constructs:
    0. DatatypeRestriction: Processes datatype restrictions (onDatatype + withRestrictions) 
       and checks against the list of XSD facets (e.g., minInclusive, maxInclusive, etc.).
    1. Enumeration (oneOf): Converts a list of individuals into a set notation.
    2. Union (unionOf): Converts a union of class expressions into a disjunction.
    3. Intersection (intersectionOf): Converts an intersection of class expressions into a conjunction.
    4. Complement (complementOf): Converts a complement of a class expression into a negation.
    5. Restriction: Processes property restrictions, including:
        - Qualified cardinalities (qualifiedCardinality, minQualifiedCardinality, maxQualifiedCardinality).
        - Value restrictions (hasValue, someValuesFrom, allValuesFrom).
        - Unqualified cardinalities (cardinality, minCardinality, maxCardinality).
    6. URI, QName, or label: Converts the node into a human-readable format using `format_node`.

    Notes:
        - The function uses helper functions like `parse_rdf_list` and `format_node` to process RDF lists 
          and format nodes, respectively.
        - If a construct is not recognized, the function defaults to formatting the node as a URI, QName, or label.
    """
    # 0. DatatypeRestriction processing (onDatatype + withRestrictions)
    on_dt = g.value(node, OWL.onDatatype)
    wr   = g.value(node, OWL.withRestrictions)
    if on_dt is not None and wr is not None:
        # restriction processing
        facets = []
        for rnode in parse_rdf_list(wr, g):
            for facet_prop in [XSD.minInclusive, XSD.maxInclusive, XSD.minExclusive, XSD.maxExclusive]:
                val = g.value(rnode, facet_prop)
                if val is not None:
                    facet_name = facet_prop.split('#')[-1]
                    facets.append(f"{facet_name} {val}")
        dt_qname = format_node(on_dt, g, label_map)
        return f"DatatypeRestriction({dt_qname} {' '.join(facets)})"

    # 1. enumeration (oneOf)
    one_of = g.value(node, OWL.oneOf)
    if one_of is not None:
        items = parse_rdf_list(one_of, g)
        return "{" + ", ".join(format_node(x, g, label_map) for x in items) + "}"

    # 2. union
    union_list = g.value(node, OWL.unionOf)
    if union_list is not None:
        items = parse_rdf_list(union_list, g)
        return "(" + " or ".join(process_class_expression(x, g, label_map) for x in items) + ")"

    # 3. intersection
    inter_list = g.value(node, OWL.intersectionOf)
    if inter_list is not None:
        items = parse_rdf_list(inter_list, g)
        return "(" + " and ".join(process_class_expression(x, g, label_map) for x in items) + ")"

    # 4. complement
    comp = g.value(node, OWL.complementOf)
    if comp is not None:
        return "not " + process_class_expression(comp, g, label_map)

    # 5. restriction
    if (node, RDF.type, OWL.Restriction) in g:
        # onProperty processing
        on_prop = g.value(node, OWL.onProperty)
        if isinstance(on_prop, URIRef):
            prop_str = format_node(on_prop, g, label_map)
        elif isinstance(on_prop, BNode):
            inv = g.value(on_prop, OWL.inverseOf)
            prop_str = "inverseOf " + format_node(inv, g, label_map) if isinstance(inv, URIRef) else format_node(on_prop, g, label_map)
        else:
            prop_str = "?"

        # 5.1 qualifiedCardinality / minQualified / maxQualified
        q_card    = g.value(node, OWL.qualifiedCardinality)
        min_qcard = g.value(node, OWL.minQualifiedCardinality)
        max_qcard = g.value(node, OWL.maxQualifiedCardinality)
        if q_card or min_qcard or max_qcard:
            if q_card:
                label_card, count = "exactly", q_card
            elif min_qcard:
                label_card, count = "min", min_qcard
            else:
                label_card, count = "max", max_qcard
            filler = g.value(node, OWL.onDataRange) or g.value(node, OWL.onClass)
            if filler is not None:
                filler_str = process_class_expression(filler, g, label_map)
                return f"[{prop_str} {label_card} {count} {filler_str}]"
            else:
                return f"[{prop_str} {label_card} {count}]"

        # 5.2 hasValue / someValuesFrom / allValuesFrom
        for pred, label in [
            (OWL.hasValue,       "hasValue"),
            (OWL.someValuesFrom, "some"),
            (OWL.allValuesFrom,  "only"),
        ]:
            val = g.value(node, pred)
            if val is not None:
                return f"[{prop_str} {label} {process_class_expression(val, g, label_map)}]"

        # 5.3 unqualified cardinalities
        for pred, label in [
            (OWL.cardinality,    "exactly"),
            (OWL.minCardinality, "min"),
            (OWL.maxCardinality, "max"),
        ]:
            val = g.value(node, pred)
            if val is not None:
                return f"[{prop_str} {label} {val}]"

        return f"[{prop_str} ?]"

    # 6.  (URI, QName, label)
    return format_node(node, g, label_map)



def extract_for_file(file_path, file_format=None):
    g = Graph()
    if not file_format:
        file_format = "turtle" if file_path.endswith(".ttl") else "xml"
    try:
        g.parse(file_path, format=file_format)
    except Exception as e:
        print(f"❌ failed ({file_path}): {e}")
        return

    # label map
    use_label = os.path.basename(file_path) in TARGET_WITH_LABEL
    label_map = {}
    if use_label:
        for s, _, lbl in g.triples((None, RDFS.label, None)):
            label_map[s] = str(lbl)

    # file name processing
    base = os.path.splitext(os.path.basename(file_path))[0]
    axioms_file = os.path.join(AXIOM_DIR, f"{base}_axiom.json")


    # Class Axioms
    raw_cls = {"subclass": [], "disjoint": [], "equivalent": [], "restriction": []}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(o, rdflib.term.BNode) and (o, RDF.type, OWL.Restriction) in g:
            raw_cls["restriction"].append((format_node(s, g, label_map), process_class_expression(o, g, label_map)))
        else:
            raw_cls["subclass"].append((format_node(s, g, label_map), process_class_expression(o, g, label_map)))
    
    # DisjointWith
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        raw_cls["disjoint"].append((
            process_class_expression(s, g, label_map),
            process_class_expression(o, g, label_map)
        ))

    # EquivalentClass 
    for s, _, o in g.triples((None, OWL.equivalentClass, None)):
        subj_expr = process_class_expression(s, g, label_map)
        obj_expr  = process_class_expression(o, g, label_map)
        raw_cls["equivalent"].append((subj_expr, obj_expr))

    class_axioms = {}
    map_cls = {"subclass": "subClassOf", "disjoint": "disjointWith", "equivalent": "equivalentClass", "restriction": "propertyRestrictions"}
    for key, rel in map_cls.items():
        for subj, expr in raw_cls[key]:
            entry = class_axioms.setdefault(subj, {})
            entry.setdefault(rel, []).append(expr)

    # Property Axioms
    props = set(g.subjects(RDF.type, OWL.ObjectProperty)) | set(g.subjects(RDF.type, OWL.DatatypeProperty))
    prop_axioms = {}
    for prop in props:
        pstr = format_node(prop, g, label_map)
        types = [lab for cls, lab in [(OWL.ObjectProperty, "ObjectProperty"), (OWL.DatatypeProperty, "DatatypeProperty")] if (prop, RDF.type, cls) in g]
        chars = [lab for cls, lab in [(OWL.FunctionalProperty, "Functional"), (OWL.InverseFunctionalProperty, "InverseFunctional"),
                                      (OWL.TransitiveProperty, "Transitive"), (OWL.SymmetricProperty, "Symmetric"),
                                      (OWL.AsymmetricProperty, "Asymmetric"), (OWL.ReflexiveProperty, "Reflexive"),
                                      (OWL.IrreflexiveProperty, "Irreflexive")] if (prop, RDF.type, cls) in g]
        doms = [process_class_expression(d, g, label_map) for d in g.objects(prop, RDFS.domain)]
        if not doms:
            doms = ["None"]
        rngs = [process_class_expression(r, g, label_map) for r in g.objects(prop, RDFS.range)]
        if not rngs:
            rngs = ["None"]
        supers = [format_node(o, g, label_map) for o in g.objects(prop, RDFS.subPropertyOf)]
        inverses = [format_node(o, g, label_map) for o in g.objects(prop, OWL.inverseOf)]
        prop_axioms[pstr] = {
            "characteristics": chars,
            "domain": doms,
            "range": rngs,
            "subPropertyOf": supers,
            "inverseOf": inverses
        }

    all_entities = [('class', k) for k in class_axioms.keys()] + \
                   [('property', k) for k in prop_axioms.keys()]
    
    if len(all_entities) > 500:
        sampled = random.sample(all_entities, 500)
        new_class_axioms = {}
        new_prop_axioms = {}
        for typ, key in sampled:
            if typ == 'class':
                new_class_axioms[key] = class_axioms[key]
            else:
                new_prop_axioms[key] = prop_axioms[key]
        class_axioms = new_class_axioms
        prop_axioms  = new_prop_axioms

    # save to file
    output = {"classes": class_axioms, "properties": prop_axioms}
    with open(axioms_file, "w", encoding="utf-8") as af:
        json.dump(output, af, indent=2, ensure_ascii=False)
    print(f"✔️ Axiom saved: {axioms_file} (classes={len(class_axioms)}, properties={len(prop_axioms)})")


def main():
    dir_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    for fname in os.listdir(dir_path):
        if fname.endswith(".owl") or fname.endswith(".ttl"):
            path = os.path.join(dir_path, fname)
            extract_for_file(path)

if __name__ == "__main__":
    main()