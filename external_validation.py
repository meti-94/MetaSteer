#!/usr/bin/env python
"""Dummy external validator.

Replace ``score_outputs`` with task-specific scoring while keeping the command
line contract intact. The checkpoint is loaded by vLLM as either a full model
or a LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


def _load_engine(checkpoint: Path, vllm_cfg: dict[str, Any]):
    adapter_config = checkpoint / "adapter_config.json"
    common = {
        "tensor_parallel_size": int(vllm_cfg.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(vllm_cfg.get("gpu_memory_utilization", 0.8)),
        "trust_remote_code": bool(vllm_cfg.get("trust_remote_code", False)),
    }
    if vllm_cfg.get("max_model_len") is not None:
        common["max_model_len"] = int(vllm_cfg["max_model_len"])

    if adapter_config.exists():
        from vllm.lora.request import LoRARequest

        adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model = adapter["base_model_name_or_path"]
        engine = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=int(adapter["r"]),
            **common,
        )
        return engine, LoRARequest("dpo-checkpoint", 1, str(checkpoint)), base_model

    engine = LLM(model=str(checkpoint), **common)
    return engine, None, str(checkpoint)


def _as_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, dict):
        if "role" not in prompt or "content" not in prompt:
            raise ValueError(f"Prompt dict must include role/content: {prompt!r}")
        return [{"role": str(prompt["role"]), "content": str(prompt["content"])}]
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError(f"Prompt message must include role/content: {message!r}")
            messages.append({"role": str(message["role"]), "content": str(message["content"])})
        if not messages:
            raise ValueError("Prompt message list is empty.")
        return messages
    return [{"role": "user", "content": str(prompt)}]


def _render_prompts(prompts: list[Any], tokenizer_name: str, trust_remote_code: bool) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    rendered: list[str] = []
    for prompt in prompts:
        rendered.append(
            tokenizer.apply_chat_template(
                _as_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


def score_outputs(texts: list[str]) -> dict[str, float]:
    """Placeholder metric to prove that periodic evaluation is wired correctly."""
    lengths = [len(text) for text in texts]
    return {
        "dummy_mean_output_characters": sum(lengths) / max(len(lengths), 1),
        "dummy_nonempty_fraction": sum(bool(text.strip()) for text in texts) / max(len(texts), 1),
    }


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    config = json.loads(args.config_json)
    prompts = config.get(
        "prompts",
        [
            "Write one sentence explaining why preference learning is useful.",
            "Give a concise and helpful greeting.",
        ],
    )
    vllm_cfg = config.get("vllm", {})

    engine, lora_request, loaded_model = _load_engine(checkpoint, vllm_cfg)
    rendered_prompts = _render_prompts(
        prompts,
        tokenizer_name=loaded_model,
        trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
    )
    sampling_cfg = config.get("sampling", {})
    sampling = SamplingParams(
        temperature=float(sampling_cfg.get("temperature", 0.0)),
        top_p=float(sampling_cfg.get("top_p", 1.0)),
        max_tokens=int(sampling_cfg.get("max_tokens", 128)),
    )
    outputs = engine.generate(rendered_prompts, sampling, lora_request=lora_request)
    texts = [output.outputs[0].text for output in outputs]

    result = {
        "checkpoint": str(checkpoint),
        "loaded_model": loaded_model,
        "metrics": score_outputs(texts),
        "samples": [
            {
                "prompt": prompt,
                "rendered_prompt": rendered,
                "output": text,
            }
            for prompt, rendered, text in zip(prompts, rendered_prompts, texts)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"external_validation": result["metrics"]}))


if __name__ == "__main__":
    main()
