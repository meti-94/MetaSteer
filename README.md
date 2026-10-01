# MetaSteer

DPO training and data preparation for concept-conditioned steering adapters.

This repository includes **code, configs, and six trained LoRA adapters** under `checkpoints/`. No separate adapter download is required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gated base models (Gemma, Llama) require a Hugging Face account and `huggingface-cli login`. Public base models (Qwen) and the Concept16k datasets also use the Hub for downloads during setup or data prep.

## Steering checkpoints

Adapters are LoRA weights only. Load them on top of the matching base model. Paths are relative to the repo root.


| Base model                         | Config                            | Checkpoint                       |
| ---------------------------------- | --------------------------------- | -------------------------------- |
| `Qwen/Qwen3-1.7B`                  | `configs/dpo_qwen3_1.7b.yaml`     | `checkpoints/dpo-qwen3-1.7b`     |
| `Qwen/Qwen3-4B`                    | `configs/dpo_qwen3_4b.yaml`       | `checkpoints/dpo-qwen3-4b`       |
| `google/gemma-2-2b-it`             | `configs/dpo_gemma2_2b_it.yaml`   | `checkpoints/dpo-gemma2-2b-it`   |
| `google/gemma-2-9b-it`             | `configs/dpo_gemma2_9b_it.yaml`   | `checkpoints/dpo-gemma2-9b-it`   |
| `meta-llama/Llama-3.2-3B-Instruct` | `configs/dpo_llama3.2_3b_it.yaml` | `checkpoints/dpo-llama3.2-3b-it` |
| `meta-llama/Llama-3.1-8B-Instruct` | `configs/dpo_llama3.1_8b_it.yaml` | `checkpoints/dpo-llama3.2-8b-it` |


Each checkpoint directory contains `adapter_config.json`, `adapter_model.safetensors`, and tokenizer files needed for PEFT loading.

### Run steered inference

Prompts use the same template as training:

```text
### Instruction
{question}

### Steering Direction
{direction}
```

```bash
python inference.py \
  --config configs/dpo_qwen3_1.7b.yaml \
  --checkpoint-path ./checkpoints/dpo-qwen3-1.7b \
  --question "How to lose weight?" \
  --direction "Procedural instructions related to software or game installation"
```

Minimal PEFT load:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3-1.7B"
adapter = "./checkpoints/dpo-qwen3-1.7b"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```



## Data download and preparation

Preference JSONL files are **not** committed. Rebuild them with the pipeline below.

### Sources


| Dataset       | Hub ID                         | Used by              |
| ------------- | ------------------------------ | -------------------- |
| Concept16k v1 | `pyvene/axbench-concept16k`    | Qwen + Llama configs |
| Concept16k v2 | `pyvene/axbench-concept16k_v2` | Gemma configs        |




### One-shot pipeline

```bash
bash data_preparation.sh
```

This will:

1. Download Concept16k v1/v2 (`data_raw/download.py`)
2. Convert them into preference JSONL under `training_ready_data/data/`
3. Delete raw downloads to free disk (skip cleanup with `KEEP_RAW=1 bash data_preparation.sh`)

Outputs:

- `training_ready_data/data/concept16k_v1_{train,valid,test}.jsonl`
- `training_ready_data/data/concept16k_v2_{train,valid,test}.jsonl`



### Step-by-step (if you prefer)

```bash
# Concept16k dumps (optional for v1: concept16k_v1.py can pull parquet from the Hub directly)
python data_raw/download.py

# Build training-ready preference pairs
python training_ready_data/concept16k_v1.py
python training_ready_data/concept16k_v2.py
```



### Disk notes

Set `HF_HOME` to a large volume if your home quota is tight: `export HF_HOME=/path/to/large/cache`.

## Training

After data preparation:

```bash
python dpo_train.py --config configs/dpo_qwen3_1.7b.yaml
```

Training writes under `./checkpoints/<experiment-name>/` (see each YAML `training.output_dir`). The shipped adapters live in those same directories; change `training.output_dir` if you want to retrain without overwriting them.

W&B logging is optional; edit `experiment.report_to` / `wandb_*` in the config if needed.

`external_validation.py` is invoked automatically when `external_validation.enabled: true` in a config.

## Repository layout

```text
├── dpo_train.py                 # DPO + LoRA training
├── inference.py                 # steered generation with a LoRA adapter
├── external_validation.py       # optional mid-training validation helper
├── data_preparation.sh          # end-to-end data pipeline
├── checkpoints/                 # shipped final LoRA adapters (6 models)
│   ├── dpo-qwen3-1.7b/
│   ├── dpo-qwen3-4b/
│   ├── dpo-gemma2-2b-it/
│   ├── dpo-gemma2-9b-it/
│   ├── dpo-llama3.2-3b-it/
│   └── dpo-llama3.2-8b-it/
├── data_raw/
│   └── download.py              # Concept16k v1/v2
├── training_ready_data/
│   ├── concept16k_v1.py
│   └── concept16k_v2.py
└── configs/                     # one YAML per base model
```



## License

Respect the licenses of the base models (Qwen, Gemma, Llama) and source datasets when redistributing adapters or derived data.