#!/usr/bin/env python
"""Config-driven conversational DPO training with periodic external evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


ROOT = Path(__file__).resolve().parent

STEERING_HEADER = "### Steering Direction\n"
SUPPRESSING_HEADER = "### Suppressing Direction\n"


def _path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _dtype(name: str) -> torch.dtype:
    dtypes = {
        "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported model.dtype {name!r}; choose one of {list(dtypes)}")
    return dtypes[name]


def _cfg_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present config key (supports short and legacy names)."""
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _load_file(path: Path, data_format: str | None) -> Dataset:
    fmt = data_format
    if fmt is None:
        fmt = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}.get(path.suffix)
    if fmt is None:
        raise ValueError(f"Cannot infer the data format for {path}; set data.format in the config.")
    return load_dataset(fmt, data_files=str(path), split="train")


def _load_split(data_cfg: dict[str, Any], split: str) -> Dataset | None:
    local_path = _path(data_cfg.get(f"{split}_path"))
    hub_name = data_cfg.get("dataset_name")
    hub_split = data_cfg.get(f"{split}_split")

    if local_path is not None:
        return _load_file(local_path, data_cfg.get("format"))
    if hub_name and hub_split:
        return load_dataset(hub_name, data_cfg.get("dataset_config"), split=hub_split)
    return None


def _validate_conversational(dataset: Dataset, split: str, columns: dict[str, str]) -> Dataset:
    missing = [name for name in columns.values() if name not in dataset.column_names]
    if missing:
        raise ValueError(f"{split} data is missing columns: {missing}")

    def check_messages(value: Any, column: str) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{split}.{column} must be a non-empty list of role/content messages.")
        for message in value:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError(f"{split}.{column} contains a message without role/content: {message!r}")

    for example in dataset.select(range(min(20, len(dataset)))):
        for canonical, source in columns.items():
            check_messages(example[source], source)

    if columns != {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}:
        reverse = {source: canonical for canonical, source in columns.items()}
        dataset = dataset.rename_columns(reverse)
    return dataset


def _limit(dataset: Dataset | None, count: int | None, seed: int) -> Dataset | None:
    if dataset is None or count is None or count <= 0 or count >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(count))


def _apply_reverse(dataset: Dataset) -> Dataset:
    """Train for suppression by swapping preferences and relabeling the prompt."""

    def transform(example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        for message in example["prompt"]:
            content = str(message["content"]).replace(STEERING_HEADER, SUPPRESSING_HEADER)
            prompt.append({"role": message["role"], "content": content})
        return {
            "prompt": prompt,
            "chosen": example["rejected"],
            "rejected": example["chosen"],
        }

    return dataset.map(transform, desc="Applying reverse (suppression) transform")


class ExperimentCallback(TrainerCallback):
    def __init__(
        self,
        result_file: Path,
        external_cfg: dict[str, Any],
        experiment_name: str,
    ) -> None:
        self.result_file = result_file
        self.external_cfg = external_cfg
        self.experiment_name = experiment_name
        self.result_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": self.experiment_name,
            **record,
        }
        with self.result_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            self._write({"type": "trainer", "step": state.global_step, "metrics": logs})

    def on_save(self, args, state, control, **kwargs):
        cfg = self.external_cfg
        interval = int(cfg.get("every_n_steps", 0))
        if not state.is_world_process_zero or not cfg.get("enabled", False) or interval <= 0:
            return
        if state.global_step == 0 or state.global_step % interval:
            return

        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        output_path = self.result_file.parent / "external" / f"step-{state.global_step}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        script = _path(cfg.get("script", "external_validation.py"))
        if script is None:
            raise ValueError("external_validation.script is required.")

        command = [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_path),
            "--config-json",
            json.dumps(cfg),
        ]
        env = os.environ.copy()
        if cfg.get("cuda_visible_devices") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])

        try:
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                capture_output=True,
                timeout=int(cfg.get("timeout_seconds", 1800)),
                check=True,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            metrics = {f"external/{key}": value for key, value in result.get("metrics", {}).items()}
            self._write(
                {
                    "type": "external_validation",
                    "step": state.global_step,
                    "checkpoint": str(checkpoint),
                    "metrics": metrics,
                    "output": str(output_path),
                }
            )
            if "wandb" in args.report_to and metrics:
                import wandb

                wandb.log(metrics, step=state.global_step)
            if completed.stdout:
                print(completed.stdout.rstrip())
        except Exception as error:
            self._write(
                {
                    "type": "external_validation_error",
                    "step": state.global_step,
                    "checkpoint": str(checkpoint),
                    "error": str(error),
                }
            )
            print(f"External validation failed at step {state.global_step}: {error}", file=sys.stderr)
            if cfg.get("fail_training_on_error", False):
                raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Train for suppression instead of steering: swap chosen/rejected and "
            "replace '### Steering Direction' with '### Suppressing Direction' in prompts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    cli = _parse_args()
    config_path = Path(cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    experiment = config["experiment"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    lora_cfg = config.get("lora", {})
    external_cfg = config.get("external_validation", {})
    name = experiment["name"]
    if cli.reverse:
        name = f"{name}-reverse"
    seed = int(experiment.get("seed", 42))

    output_dir = _path(train_cfg["output_dir"])
    if cli.reverse and output_dir is not None:
        output_dir = Path(f"{output_dir}-reverse")
    results_dir = _path(experiment.get("results_dir", "results"))
    assert output_dir is not None and results_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    run_results_dir = results_dir / name
    run_results_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_results_dir / "metrics.jsonl"
    saved_config = {**config, "cli": {"reverse": cli.reverse}}
    (run_results_dir / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False),
        encoding="utf-8",
    )

    epochs = float(_cfg_get(train_cfg, "epochs", "num_train_epochs", default=1))
    batch_size = int(_cfg_get(train_cfg, "batch_size", "per_device_train_batch_size", default=4))
    eval_batch_size = int(_cfg_get(train_cfg, "eval_batch_size", "per_device_eval_batch_size", default=batch_size))
    grad_accum = int(
        _cfg_get(train_cfg, "gradient_accumulation", "gradient_accumulation_steps", default=8)
    )
    save_steps = int(_cfg_get(train_cfg, "save_nsteps", "save_steps", default=200))
    validation_steps = int(_cfg_get(train_cfg, "validation_nsteps", "eval_steps", default=save_steps))
    lora_rank = int(_cfg_get(lora_cfg, "rank", "r", default=32))

    interval = int(external_cfg.get("every_n_steps", 0))
    if external_cfg.get("enabled", False) and (interval <= 0 or interval % save_steps != 0):
        raise ValueError(
            "external_validation.every_n_steps must be a positive multiple of training.save_nsteps "
            "so a checkpoint exists at every external validation step."
        )

    columns = data_cfg.get("columns", {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"})
    train_dataset = _load_split(data_cfg, "train")
    validation_dataset = _load_split(data_cfg, "validation")
    test_dataset = _load_split(data_cfg, "test")
    if train_dataset is None:
        raise ValueError("Configure data.train_path or data.dataset_name + data.train_split.")

    train_dataset = _validate_conversational(train_dataset, "train", columns)
    if validation_dataset is not None:
        validation_dataset = _validate_conversational(validation_dataset, "validation", columns)
    if test_dataset is not None:
        test_dataset = _validate_conversational(test_dataset, "test", columns)
    validation_dataset = _limit(validation_dataset, data_cfg.get("validation_max_samples"), seed)
    test_dataset = _limit(test_dataset, data_cfg.get("test_max_samples"), seed)

    if cli.reverse:
        print(
            "Reverse mode enabled: swapping chosen/rejected and using "
            "'### Suppressing Direction' in prompts."
        )
        train_dataset = _apply_reverse(train_dataset)
        if validation_dataset is not None:
            validation_dataset = _apply_reverse(validation_dataset)
        if test_dataset is not None:
            test_dataset = _apply_reverse(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        # Prefer a dedicated pad token when the tokenizer provides one so EOS is
        # not masked during DPO loss (same convention as DPO.py).
        vocab = tokenizer.get_vocab()
        if "<|finetune_right_pad_id|>" in vocab:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    report_to = experiment.get("report_to", [])
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" in report_to:
        os.environ["WANDB_PROJECT"] = experiment.get("wandb_project", "DPO_Steering")
        if experiment.get("wandb_entity"):
            os.environ["WANDB_ENTITY"] = experiment["wandb_entity"]

    dtype = _dtype(model_cfg.get("dtype", "auto"))
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        run_name=name,
        seed=seed,
        report_to=report_to,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        beta=float(train_cfg["beta"]),
        max_length=int(train_cfg["max_length"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit"),
        eval_strategy="steps" if validation_dataset is not None else "no",
        eval_steps=validation_steps if validation_dataset is not None else None,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        remove_unused_columns=False,
        model_init_kwargs={
            "dtype": dtype,
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        },
    )

    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg["target_modules"],
        )

    callback = ExperimentCallback(result_file, external_cfg, name)
    trainer = DPOTrainer(
        model=model_cfg["name_or_path"],
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )
    train_result = trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir / "final"))
    callback._write({"type": "train_complete", "step": trainer.state.global_step, "metrics": train_result.metrics})

    if test_dataset is not None:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        callback._write({"type": "test", "step": trainer.state.global_step, "metrics": test_metrics})


if __name__ == "__main__":
    main()
