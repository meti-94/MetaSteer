#!/usr/bin/env python
"""Build DPO-ready preference data from the SteerBoost result dumps.

Walks data_raw/data/steerboost/result/{method}/{model}/{concept_id}/ and reads
the {layer}-{alpha}.json files one at a time (memory friendly: only one
concept directory is held in memory at once). Generations are grouped by
instruction; the steering concept text comes from concepts.json via the
concept_id directory name.

Judgment labels: 1 = successful steering (chosen), 0 = under-steered and
2 = over-steered (both rejected). For every (method, model, concept,
instruction) group one preference pair is built: the first successful
generation as chosen and a randomly sampled unsuccessful one as rejected.

Split follows the concept16k approach: by concept only (75/10/15, seed 42),
so every concept appears in exactly one of train/valid/test.

Output: steerboost_{train,valid,test}.jsonl under training_ready_data/data/.
"""

import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STEERBOOST_DIR = os.path.join(BASE_DIR, "..", "data_raw", "data", "steerboost")
RESULT_DIR = os.path.join(STEERBOOST_DIR, "result")
CONCEPTS_PATH = os.path.join(STEERBOOST_DIR, "concepts.json")
OUT_DIR = os.path.join(BASE_DIR, "data")
OUT_PREFIX = "steerboost"

PROMPT_TEMPLATE = (
    "### Instruction\n"
    "{instruction}\n\n"
    "### Steering Direction\n"
    "{direction}"
)

MIN_LEN = 10
CHOSEN_LABEL = 1  # judgment 1 = successful steering; 0/2 = rejected
SAMPLE_RANDOM_SEED = 44
SPLIT_RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.75, 0.10  # remainder goes to test


def load_concept_names() -> dict:
    with open(CONCEPTS_PATH, encoding="utf-8") as f:
        concepts = json.load(f)
    return {c["concept_id"]: c["concept_name"] for c in concepts}


def iter_concept_dirs():
    """Yield (method, model, concept_id, dir_path) for every concept directory."""
    for method in sorted(os.listdir(RESULT_DIR)):
        method_dir = os.path.join(RESULT_DIR, method)
        if not os.path.isdir(method_dir):
            continue
        for model in sorted(os.listdir(method_dir)):
            model_dir = os.path.join(method_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for concept_id in sorted(os.listdir(model_dir), key=int):
                concept_dir = os.path.join(model_dir, concept_id)
                if os.path.isdir(concept_dir):
                    yield method, model, int(concept_id), concept_dir


def build_pairs(concept_names: dict, rng: np.random.Generator):
    prompts, accepted, rejected, steering_concepts = [], [], [], []
    concept_dirs = list(iter_concept_dirs())

    for method, model, concept_id, concept_dir in tqdm(
        concept_dirs, desc="Building pairs"
    ):
        concept_name = concept_names.get(concept_id)
        if concept_name is None:
            print(f"[warn] concept_id {concept_id} not in concepts.json, skipping")
            continue

        # Group all generations of this (method, model, concept) by instruction.
        chosen_by_inst = defaultdict(list)
        rejected_by_inst = defaultdict(list)
        for fname in sorted(os.listdir(concept_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(concept_dir, fname), encoding="utf-8") as f:
                payload = json.load(f)
            for sample in payload.get("results", []):
                response = sample.get("response_after")
                if not response or len(response) < MIN_LEN:
                    continue
                instruction = sample.get("instruction")
                if not instruction:
                    continue
                if sample.get("judgment") == CHOSEN_LABEL:
                    chosen_by_inst[instruction].append(response)
                else:
                    rejected_by_inst[instruction].append(response)

        # One pair per instruction: first successful response vs a randomly
        # sampled unsuccessful one (mirrors the concept16k pairing).
        for instruction in sorted(chosen_by_inst):
            candidates = rejected_by_inst.get(instruction)
            if not candidates:
                continue
            temp_prompt = PROMPT_TEMPLATE.format(
                instruction=instruction, direction=concept_name
            )
            if len(temp_prompt) < MIN_LEN:
                continue
            prompts.append(temp_prompt)
            accepted.append(chosen_by_inst[instruction][0])
            rejected.append(candidates[rng.integers(len(candidates))])
            steering_concepts.append(concept_name)

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
    # one of train/valid/test (same approach as concept16k).
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
    concept_names = load_concept_names()
    print(f"Loaded {len(concept_names)} concepts from {CONCEPTS_PATH}")
    rng = np.random.default_rng(SAMPLE_RANDOM_SEED)
    prompts, accepted, rejected, steering_concepts = build_pairs(concept_names, rng)
    split_and_save(prompts, accepted, rejected, steering_concepts)
    print("Done.")


if __name__ == "__main__":
    main()
