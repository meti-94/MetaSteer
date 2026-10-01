#!/usr/bin/env python
"""Run a single DPO LoRA checkpoint inference with a question and direction.

Builds the same user prompt used during DPO training:

    ### Instruction
    {instruction}

    ### Steering Direction
    {direction}

With ``--reverse``, uses the suppression template and looks for the
``{output_dir}-reverse`` checkpoint written by ``dpo_train.py --reverse``:

    ### Instruction
    {instruction}

    ### Suppressing Direction
    {direction}

Example:
    python inference.py \\
        --config configs/dpo_gemma3_4b.yaml \\
        --checkpoint 846 \\
        --question "Should I accept this ultimatum?" \\
        --direction "Be more cooperative and willing to compromise."

    python inference.py \\
        --config configs/dpo_gemma3_4b.yaml \\
        --checkpoint 200 \\
        --reverse \\
        --question "Should I accept this ultimatum?" \\
        --direction "Be more cooperative and willing to compromise."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent

STEERING_PROMPT_TEMPLATE = (
    "### Instruction\n"
    "{instruction}\n\n"
    "### Steering Direction\n"
    "{direction}"
)

SUPPRESSING_PROMPT_TEMPLATE = (
    "### Instruction\n"
    "{instruction}\n\n"
    "### Suppressing Direction\n"
    "{direction}"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one steered (or suppressed) response with a DPO LoRA checkpoint."
    )
    parser.add_argument("--config", required=True, help="Training YAML configuration.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=int,
        help="Checkpoint step under training.output_dir, e.g. 200. Ignored if --checkpoint-path is set.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit LoRA checkpoint directory (overrides --checkpoint and output_dir).",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="User question / instruction. Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--direction",
        default=None,
        help="Steering/suppressing direction / concept. Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Use suppression mode: '### Suppressing Direction' in the prompt and "
            "load from {output_dir}-reverse when resolving --checkpoint "
            "(matching dpo_train.py --reverse)."
        ),
    )
    args = parser.parse_args()
    if args.checkpoint_path is None and args.checkpoint is None:
        parser.error("Provide --checkpoint STEP or --checkpoint-path DIR.")
    return args


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _dtype(name: str) -> torch.dtype:
    dtypes = {
        "auto": torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported model.dtype {name!r}; choose one of {list(dtypes)}")
    return dtypes[name]


def _checkpoint_metadata(checkpoint: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    adapter_path = checkpoint / "adapter_config.json"
    if not adapter_path.is_file():
        raise FileNotFoundError(
            f"{adapter_path} was not found. This inference script expects a LoRA checkpoint."
        )

    weight_files = list(checkpoint.glob("adapter_model*.safetensors")) + list(
        checkpoint.glob("adapter_model*.bin")
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No adapter_model weights were found in {checkpoint}. "
            "The checkpoint may still be saving or may be incomplete."
        )

    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    expected_rank = config.get("lora", {}).get("rank", config.get("lora", {}).get("r"))
    if expected_rank is not None and int(expected_rank) != int(adapter["r"]):
        raise ValueError(
            f"Config LoRA rank is {expected_rank}, but checkpoint rank is {adapter['r']}. "
            "Use the training config corresponding to this checkpoint."
        )

    configured_model = config["model"]["name_or_path"]
    checkpoint_model = adapter["base_model_name_or_path"]
    if configured_model != checkpoint_model:
        raise ValueError(
            f"Config base model is {configured_model!r}, but checkpoint expects "
            f"{checkpoint_model!r}. Use the matching training config."
        )
    return adapter


def _build_user_prompt(instruction: str, direction: str, *, reverse: bool = False) -> str:
    template = SUPPRESSING_PROMPT_TEMPLATE if reverse else STEERING_PROMPT_TEMPLATE
    return template.format(
        instruction=instruction.strip(),
        direction=direction.strip(),
    )


def _supports_system_role(model_id: str | None) -> bool:
    """Gemma chat templates only allow user/model turns (no system role)."""
    if not model_id:
        return True
    return "gemma" not in model_id.lower()


def _is_qwen_family(model_id: str | None) -> bool:
    if not model_id:
        return False
    return "qwen" in model_id.lower()


def _resolve_enable_thinking(
    enable_thinking: bool | None,
    config: dict[str, Any] | None,
    model_id: str | None,
) -> bool:
    """CLI override > YAML inference.enable_thinking > Qwen-family default True."""
    if enable_thinking is not None:
        return bool(enable_thinking)
    inference = (config or {}).get("inference") or {}
    if "enable_thinking" in inference:
        return bool(inference["enable_thinking"])
    return _is_qwen_family(model_id)


def _build_messages(
    user_prompt: str,
    system_prompt: str | None,
    *,
    supports_system_role: bool = True,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    content = user_prompt
    if system_prompt and supports_system_role:
        messages.append({"role": "system", "content": system_prompt})
    elif system_prompt:
        # Gemma rejects role=system; keep the text by folding into the user turn.
        content = f"{system_prompt}\n\n{user_prompt}"
    messages.append({"role": "user", "content": content})
    return messages


def _prepare_tokenizer(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        vocab = tokenizer.get_vocab()
        if "<|finetune_right_pad_id|>" in vocab:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def _load_model(
    base_model: str,
    checkpoint: Path,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> PeftModel:
    print(f"Loading base model {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    print(f"Loading LoRA adapter from {checkpoint}")
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()
    return model


def _apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None = None,
    **extra: Any,
):
    """apply_chat_template with optional Qwen3 enable_thinking support."""
    kwargs = {"add_generation_prompt": True, **extra}
    if enable_thinking is None:
        return tokenizer.apply_chat_template(messages, **kwargs)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": enable_thinking},
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)


def _tokenize_messages(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    *,
    enable_thinking: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Normalize apply_chat_template output to an input_ids / attention_mask dict.

    Depending on transformers version, apply_chat_template(tokenize=True,
    return_tensors="pt") may return a BatchEncoding, a bare Tensor, or a list.
    Pass ``enable_thinking`` for Qwen3 chat templates.
    """
    encoded = _apply_chat_template(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
    elif isinstance(encoded, dict) or hasattr(encoded, "keys"):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            return {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }
    else:
        input_ids = torch.tensor(encoded, dtype=torch.long)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def _generate_response(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    sampling_cfg: dict[str, Any],
    *,
    enable_thinking: bool | None = None,
) -> str:
    device = model.get_input_embeddings().weight.device
    inputs = _tokenize_messages(
        tokenizer, messages, device, enable_thinking=enable_thinking
    )
    input_len = inputs["input_ids"].shape[-1]

    temperature = float(sampling_cfg.get("temperature", 0.7))
    top_p = float(sampling_cfg.get("top_p", 0.9))
    max_new_tokens = int(sampling_cfg.get("max_tokens", 512))
    repetition_penalty = float(sampling_cfg.get("repetition_penalty", 1.0))

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    print(f"gen_kwargs: {gen_kwargs}")
    print("--------------------------------")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0, input_len:]
    print(f"new_tokens: {new_tokens}")
    print("--------------------------------")
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _read_user_input(label: str) -> str:
    value = input(f"{label}: ").strip()
    if not value:
        raise ValueError(f"{label} cannot be empty.")
    return value


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    direction_label = "Suppressing direction" if args.reverse else "Steering direction"
    question = args.question or _read_user_input("Question")
    direction = args.direction or _read_user_input(direction_label)

    output_dir = _resolve_path(config["training"]["output_dir"])
    if args.reverse:
        output_dir = Path(f"{output_dir}-reverse")
    if args.checkpoint_path is not None:
        checkpoint = _resolve_path(args.checkpoint_path)
    else:
        checkpoint = output_dir / f"checkpoint-{args.checkpoint}"
    adapter = _checkpoint_metadata(checkpoint, config)
    base_model = adapter["base_model_name_or_path"]
    inference_cfg = config.get("inference", {})
    sampling_cfg = inference_cfg.get("sampling", {})
    trust_remote_code = bool(config["model"].get("trust_remote_code", False))
    system_prompt = inference_cfg.get("system_prompt")
    dtype = _dtype(config["model"].get("dtype", "auto"))

    mode = "suppressing" if args.reverse else "steering"
    print(
        f"Mode={mode}, checkpoint={checkpoint}: LoRA rank={adapter['r']}, "
        f"alpha={adapter['lora_alpha']}, targets={sorted(adapter['target_modules'])}"
    )

    user_prompt = _build_user_prompt(question, direction, reverse=args.reverse)
    supports_system_role = _supports_system_role(base_model)
    if system_prompt and not supports_system_role:
        print(
            "Gemma family does not support a system role; "
            "folding system_prompt into the user message."
        )
    messages = _build_messages(
        user_prompt, system_prompt, supports_system_role=supports_system_role
    )

    print("\n--- Prompt ---")
    print(user_prompt)
    print("--------------\n")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    _prepare_tokenizer(tokenizer)
    model = _load_model(base_model, checkpoint, dtype, trust_remote_code)

    response = _generate_response(model, tokenizer, messages, sampling_cfg)

    print("--- Response ---")
    print(response)
    print("----------------")


if __name__ == "__main__":
    main()
