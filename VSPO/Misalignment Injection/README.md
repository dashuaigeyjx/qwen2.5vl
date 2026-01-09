# Misalignment Injection: Full Dataset Construction Pipeline


## 🧩 Step-by-Step Pipeline Overview

### 1. Generate Definitions

- **Directory**: `definition generation/`
- **Output**: `generated description/`
- **Process**: Use GPT to generate natural language definitions based on ontology axioms.


### 2. Type Classification and Splitting

- **Script**: `Type_classify.py`
- **Inputs**:
  - Generated definitions
  - Ontology axioms
  - Generated CQs
- **Output**: Dataset split by misalignment types.
- **Next Step**: Move output files to `type classification/`.


### 3. Process Each Misalignment Type

#### ➤ Type 1 & Type 2

- **Script**: `type1,2_processing.py`
- **Outputs**:
  - `Final_type1.json` → move to `processed types/`
  - `processed_type2.json` → move to `Type2 processing/` and process type 2 in `Type2 processing/` directory.

#### ➤ Type 3

- **Script**: `type3_processing.py`
- **Output**: `Final_type3.json` → move to `processed types/`

#### ➤ Type 4

- **Script**: `type4_processing.py`
- **Output**: `Final_type4.json` → move to `processed types/`


### 4. Tag Type Labels

- **Notebook**: `processed types/tagging_type.ipynb`
- **Function**: Adds misalignment type labels to each ontology term.
- **Output**: Unified data with type annotations.


### 5. Merge All Processed Data

- **Script**: `merge_data.py`
- **Outputs**:
  - `merged dataset/` → Combined dataset (all types)
  - `Final dataset/` → Training/test splits with metadata
  - `additional settings/Generalizability/unseen ontology/` → Extra dataset for unseen ontologies setting


## 📁 Key Files

```
├── Type_classify.py               # Classify and split data by misalignment type
├── type1,2_processing.py          # Process Type 1 and Type 2 entries
├── type3_processing.py            # Process Type 3 entries
├── type4_processing.py            # Process Type 4 entries
├── merge_data.py                  # Merge all processed types and split datasets
```


## 📦 Final Output (in `Final dataset/`)

- `train_dataset.jsonl` / `test_dataset.jsonl`: Final dataset for training and evaluation
- `*_meta.jsonl`: Metadata containing axiom info, original definitions, generated CQs, and assigned type

