# Template-based CQ Generation

This module generates **Competency Questions (CQs)** from OWL ontologies using axiom-specific templates and GPT API.

## 📁 Directory Overview

```
├── Axiom_per_entity/       # Output: Extracted axioms per ontology term (class/property)
├── CQ_Batchinput/          # Input batch files fed into GPT
├── CQ_Batchoutput/         # Raw GPT output for each batch
├── Generated CQ/           # Final generated CQ files
├── Ontology/               # Source ontology (.owl) files
├── templates/              # Template for each axiom type
├── CQ_generation.ipynb     # Jupyter notebook version of the CQ pipeline
├── CQ_generation.py        # Main script to generate CQs using templates + GPT
├── CQ_postprocessing.py    # Postprocesses GPT-generated CQs 
├── Ontology_processing.py  # Extracts axioms from ontologies
```

## 🔧 Workflow Description

1. **Ontology Preprocessing**

   - File: `Ontology_processing.py`
   - Function: Parses ontology files from the `Ontology/` directory, extracts relevant axioms (e.g., `subClassOf`, `inverseOf`, `subPropertyOf`, etc.) for each term.
   - Output is saved to the `Axiom_per_entity/` folder in a structured JSON format.

2. **CQ Generation**

   - File: `CQ_generation.py` (or use the notebook version)
   - Function: Uses predefined templates (in `templates/`) and the extracted axioms to create prompts.
   - Prompts are sent to GPT-based models to generate CQs.
   - Inputs and raw outputs are stored in `CQ_Batchinput/` and `CQ_Batchoutput/`, respectively.

3. **Postprocessing**

   - File: `CQ_postprocessing.py`
   - Function: Cleans and reformats the generated CQs to make them suitable for further use.
   - Final CQs are saved under `Generated CQ/`.

## ✅ Requirements

Make sure to install the dependencies:

```bash
pip install -r ../requirements.txt
```


