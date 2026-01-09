# Experiments and Evaluation


## 🧪 Main Evaluation Scripts

### 1. `evaluation.py`

- Contains evaluation functions.

### 2. `evaluate_metric.py`

- Evaluates:
  - Baseline vs. gold CQs
  - Fine-tuned model vs. gold CQs
  - GPT model vs. gold CQs
  - Generalization (unseen ontology) setting

---

## 🤖 GPT-based Generation (`GPT/` directory)

To generate and evaluate CQs using GPT-4.1:

### 1. `GPT_generation.py`

- Makes `GPT_input.jsonl` from the test_dataset.jsonl
- Sends `GPT_input.jsonl` to GPT-4.1 via API.
- Saves raw outputs to `GPT_4_1_output.jsonl`.

### 2. `gpt_output_process.py`

- Post-processes raw GPT outputs.
- Cleans and structures responses into `parsed_GPT_4_1_outputs.jsonl`.

These outputs are used for evaluation in the main script (`evaluate_metric.py`).

---

## 📁 Directory Structure

```
├── Analysis/               # Additional analysis not included in paper
├── GPT/                   # GPT-based CQ generation and outputs
├── results/               # Final evaluation outputs
├── unseen ontology/       # Inference results on unseen ontology dataset
├── vanila/                # Results from the baseline experiments(fine-tuned model and vanila Llama model)
├── test_dataset.jsonl     # Input test set for evaluation
├── test_dataset_meta.jsonl # Metadata including axiom and term descriptions
```

