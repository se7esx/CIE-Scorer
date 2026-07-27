# CIE-Scorer

CIE-Scorer implements Circuit-Guided Internal-External Discrepancy scoring for detecting unfaithful chain-of-thought (CoT) reasoning. It is set up to run unfaithfulness detection on the datasets released with [FaithCoT-BENCH](https://github.com/se7esx/FaithCoT-BENCH), including AQuA, LogiQA, TruthfulQA, and HLE-BIO.

The main entry point reads FaithCoT-BENCH `response_*.json` files, constructs sentence-level internal circuits from the provided CoT steps, compares those internal representations with external sentence embeddings, and writes an `unfaithful_score` for each example.

This repository is a cleaned implementation release. It does not include experiment logs, Slurm scripts, ablation code, private paths, generated datasets, checkpoints, model weights, or bundled `circuit-tracer` archives.

## Method

For each example, CIE-Scorer:

1. Generates a CoT response from a causal language model.
2. Splits the response into reasoning steps.
3. Builds compressed attribution circuits for the prompt and each reasoning step.
4. Converts each circuit into a PyTorch Geometric graph.
5. Encodes the external text sequence with a frozen language-model sentence encoder.
6. Computes an internal-external discrepancy score using fused Gromov-Wasserstein distance.

Larger `unfaithful_score` values indicate stronger internal-external discrepancy and therefore higher likelihood of unfaithful reasoning.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cie-scorer
```

Install `circuit-tracer` separately. This project expects a `circuit_tracer` Python package that supports `ReplacementModel` and the CRV-style transcoders:

```bash
git clone https://github.com/safety-research/circuit-tracer.git
cd circuit-tracer
pip install -e .
cd ..
```

If your experiments use gated Hugging Face models such as Llama, authenticate first:

```bash
huggingface-cli login
```

## FaithCoT-BENCH Data

Download the benchmark data from [se7esx/FaithCoT-BENCH](https://github.com/se7esx/FaithCoT-BENCH), then unzip it locally:

```bash
git clone https://github.com/se7esx/FaithCoT-BENCH.git
cd FaithCoT-BENCH
unzip faithcot.zip
```

The expected input is a directory containing files like:

```text
faithcot/
  aqua/
    llama-3.1-8b-instruct/
      response_0.json
      response_1.json
  logiqa/
  truthfulqa/
  HLE_BIO/
```

Each `response_*.json` file should include the benchmark prompt fields and a `sample_0` object with `step_*` CoT entries. If labels such as `unfaithfulness` are present, the scorer preserves them in the output for evaluation.

## Run on FaithCoT-BENCH

Score one FaithCoT-BENCH dataset/model directory:

```bash
cie-score-faithcot \
  --faithcot_dir path/to/faithcot/logiqa/llama-3.1-8b-instruct \
  --output_jsonl outputs/logiqa_llama_cie_scores.jsonl \
  --checkpoint checkpoints/cie_detector.pt \
  --work_dir work/logiqa_llama \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --transcoder_name facebook/crv-8b-instruct-transcoders \
  --device cuda
```

For other FaithCoT-BENCH subsets, change `--faithcot_dir`, for example:

```bash
cie-score-faithcot \
  --faithcot_dir path/to/faithcot/aqua/llama-3.1-8b-instruct \
  --output_jsonl outputs/aqua_llama_cie_scores.jsonl \
  --checkpoint checkpoints/cie_detector.pt \
  --work_dir work/aqua_llama \
  --device cuda
```

Cached circuits are stored under `--work_dir`, so interrupted runs can be resumed. Add `--overwrite` to rebuild them.

The output JSONL keeps benchmark metadata and adds:

- `response`
- `unfaithful_score`
- `raw_fgw_distance`
- `unfaithful_threshold`

Use `unfaithful_score > unfaithful_threshold` as the detector prediction when evaluating unfaithfulness labels.

## Generic JSONL Input

The scorer expects a JSONL file with one JSON object per line. By default, each row must contain a `prompt` field:

```json
{"idx": 0, "prompt": "Question: ..."}
{"idx": 1, "prompt": "Question: ..."}
```

Use `--prompt_field` if your prompt column has a different name.

## Checkpoint

Inference requires a trained detector checkpoint passed with `--checkpoint`. The checkpoint should contain:

- `state_dict`
- `threshold`
- `hf_model_name`
- `layer_idx`
- `model_config`
- `format_version: 1`

The helper `save_detector_checkpoint` in `cie_scorer/detector.py` writes this format.

## Run on Generic JSONL

If you want to generate fresh CoTs from prompts instead of using FaithCoT-BENCH response files, score an input JSONL file:

```bash
cie-score-jsonl \
  --input_jsonl examples/prompts.jsonl \
  --output_jsonl outputs/cie_scores.jsonl \
  --checkpoint checkpoints/cie_detector.pt \
  --work_dir work/cie_run \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --transcoder_name facebook/crv-8b-instruct-transcoders \
  --device cuda
```

Cached responses and circuits are stored under `--work_dir`, so interrupted runs can be resumed. Add `--overwrite` to regenerate them.

## Repository Layout

```text
cie_scorer/
  score_jsonl.py       # JSONL scoring CLI
  score_faithcot.py    # FaithCoT-BENCH response-directory scoring CLI
  circuit.py           # entropy-causal token selection and compressed attribution
  graph_features.py    # circuit graph to PyTorch Geometric conversion
  detector.py          # CIE detector, FGW distance, checkpoint helpers
environment.yml
pyproject.toml
README.md
```

## Citation

If you use this project, please cite:

```bibtex
@article{shen2026detecting,
  title={Detecting Unfaithful Chain-of-Thought via Circuit-Guided Internal-External Discrepancy},
  author={Shen, Xu and Tan, Zhen and Wang, Song and Hong, Pingjun and Miao, Rui and Wang, Xin and Chen, Tianlong},
  journal={arXiv preprint arXiv:2605.25603},
  year={2026}
}
```

Paper: https://arxiv.org/pdf/2605.25603

## License

Add the license that matches your release requirements before publishing.
