# LLM Fine-tuning for Next Activity Prediction

This folder contains scripts for preprocessing datasets to fine-tune Large Language Models (LLMs) for next activity prediction in business processes using [HuggingFace Autotrain](https://github.com/huggingface/autotrain-advanced?tab=readme-ov-file).

## Files

- `preprocessing_dataset.py`: Script to preprocess the S_NAP dataset into HuggingFace Autotrain format.

## S_NAP Dataset

The preprocessing script works with the S_NAP dataset from:

> A. Rebmann, F. D. Schmidt, G. Glavaš, and H. van der Aa, "Process behavior corpus and benchmarking datasets," May 2024. [Online]. Available: https://doi.org/10.5281/zenodo.11276246

### Dataset Download

To use this preprocessing script:

1. Download the S_NAP dataset from the provided DOI link
2. Place the `S_NAP.csv` file in this folder (`src/llm_sft/`)
3. Run the preprocessing script: `python3 preprocessing_dataset.py`

## Preprocessing Script

The `preprocessing_dataset.py` script converts the S_NAP dataset into the format required by HuggingFace Autotrain for fine-tuning LLMs. 

### Input Format
The script expects a CSV file with the following columns:
- `prefix`: The sequence of activities leading up to the prediction point
- `next`: The next activity to be predicted
- `unique_activities`: The set of allowed unique activity names for the process

### Output Format
The script generates a CSV file with a single `text` column in the HuggingFace Autotrain format:
```
### Human: Given the process prefix: [activity_sequence]. The allowed unique activity names are: {set_of_activities}. What is the next activity? ### Assistant: next_activity
```

### Usage
```bash
python preprocessing_dataset.py
```

The script will:
1. Look for `S_NAP.csv` in the current directory
2. Save the processed dataset as `S_NAP_autotrain.csv`
3. Display sample examples

## Fine-tuning with HuggingFace Autotrain

Once preprocessed, the `S_NAP_autotrain.csv` file can be used with [HuggingFace Autotrain](https://github.com/huggingface/autotrain-advanced?tab=readme-ov-file) for fine-tuning language models on next activity prediction tasks locally or on HuggingFace Spaces.