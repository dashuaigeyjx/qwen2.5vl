# Fine-tuning



## 🧠 Core Scripts

### 1. `train.py`

- Trains the model on the `train_dataset.jsonl` from the final dataset.
- Supports LoRA-style parameter-efficient fine-tuning.
- Training configurations (batch size, model checkpoint, logging) are customizable inside the script.

### 2. `inference.py`

- Runs the fine-tuned model on the `test_dataset.jsonl` to generate CQs.
- Output includes generated CQs and optionally the logits or confidence scores.
- Can be extended to support evaluation on different domains or unseen ontologies.


## 📁 Subdirectories

### `vanila/`

- Contains fine-tuining data and fine-tuned LoRA adapter for base experiments.

### `unseen ontology/`

- Stores fine-tuining data and fine-tuned LoRA adapter for unseen ontology setting experiments.


## 🔧 Notes

- Ensure the `train_dataset.jsonl` and `test_dataset.jsonl` from the `Misalignment Injection/Final dataset/` directory are available and properly formatted.
- Outputs from `inference.py` are used in the `Experiments/` directory for evaluation and analysis.

