#!/usr/bin/env python
"""Build DPO-ready preference data from axbench-concept16k_v2 (llama3.1/8b).

Note: the v2 repo ships three model configs (gemma/2b, gemma/9b, llama3.1/8b)
that share the same inputs. As in llama.ipynb, only the llama3.1/8b config is
used here so chosen/rejected responses come from the same model.

Pairs each (input, steering concept) with a concept-positive response (chosen)
and a sampled concept-free response (rejected), then splits by steering
concept so every concept appears in exactly one of train/valid/test.

Output: concept16k_v2_{train,valid,test}.jsonl under training_ready_data/data/.
"""

import json
import os

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "pyvene/axbench-concept16k_v2"
PARQUET_FILE = "llama3.1/8b/l20_131k/train/data.parquet"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
OUT_PREFIX = "concept16k_v2"

PROMPT_TEMPLATE = (
    "### Instruction\n"
    "{instruction}\n\n"
    "### Steering Direction\n"
    "{direction}"
)

MIN_LEN = 10
SAMPLE_RANDOM_STATE = 44
SPLIT_RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.75, 0.10  # remainder goes to test


def load_data() -> pd.DataFrame:
    # Resolves to the local HF cache (already downloaded by data_raw scripts).
    path = hf_hub_download(REPO_ID, PARQUET_FILE, repo_type="dataset")
    print(f"Loading {path} ...")
    data = pd.read_parquet(
        path, columns=["input", "output", "output_concept", "category"]
    )
    print(f"Rows loaded: {len(data):,}")
    return data


def build_pairs(data: pd.DataFrame):
    prompts, rejected, accepted, steering_concepts = [], [], [], []

    for input_prompt, input_group in tqdm(
        data.groupby("input"), total=data["input"].nunique(), desc="Building pairs"
    ):
        negative = input_group[input_group.category == "negative"]
        if len(negative) == 0:
            continue
        for output_concept, output_group in input_group.groupby("output_concept"):
            positive = output_group[output_group.category == "positive"]
            if len(positive) == 0:
                continue
            temp_prompt = PROMPT_TEMPLATE.format(
                instruction=input_prompt, direction=output_concept
            )
            temp_accepted = positive.iloc[0]["output"]
            temp_rejected = negative.sample(
                n=1, random_state=SAMPLE_RANDOM_STATE
            )["output"].iloc[0]
            if len(temp_prompt) < MIN_LEN:
                continue
            if temp_accepted is None or len(temp_accepted) < MIN_LEN:
                continue
            if temp_rejected is None or len(temp_rejected) < MIN_LEN:
                continue
            prompts.append(temp_prompt)
            rejected.append(temp_rejected)
            accepted.append(temp_accepted)
            steering_concepts.append(output_concept)

    print(f"Built {len(prompts):,} preference pairs")
    return prompts, accepted, rejected, steering_concepts


def make_record(prompt, chosen, rejected):
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def split_and_save(prompts, accepted, rejected, steering_concepts):
    # Split by steering concept only: each unique concept appears in exactly
    # one of train/valid/test.
    unique_concepts = list(dict.fromkeys(steering_concepts))
    rng = np.random.default_rng(SPLIT_RANDOM_SEED)
    shuffled = [unique_concepts[i] for i in rng.permutation(len(unique_concepts))]

    n_concepts = len(shuffled)
    n_train = int(TRAIN_FRAC * n_concepts)
    n_val = int(VAL_FRAC * n_concepts)

    train_concepts = set(shuffled[:n_train])
    val_concepts = set(shuffled[n_train : n_train + n_val])
    test_concepts = set(shuffled[n_train + n_val :])

    assert train_concepts.isdisjoint(val_concepts)
    assert train_concepts.isdisjoint(test_concepts)
    assert val_concepts.isdisjoint(test_concepts)
    assert train_concepts | val_concepts | test_concepts == set(unique_concepts)

    records = {"train": [], "valid": [], "test": []}
    for p, c, r, concept in zip(prompts, accepted, rejected, steering_concepts):
        rec = make_record(p, c, r)
        if concept in train_concepts:
            records["train"].append(rec)
        elif concept in val_concepts:
            records["valid"].append(rec)
        else:
            records["test"].append(rec)

    print(
        f"Concepts: train={len(train_concepts):,} valid={len(val_concepts):,} "
        f"test={len(test_concepts):,} (total unique={n_concepts:,})"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    for split_name, recs in records.items():
        out_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in recs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(recs):,} examples to {out_path}")


def main():
    data = load_data()
    prompts, accepted, rejected, steering_concepts = build_pairs(data)
    split_and_save(prompts, accepted, rejected, steering_concepts)
    print("Done.")


if __name__ == "__main__":
    main()
