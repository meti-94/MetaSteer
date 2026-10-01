import os

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = {
    "concept16k": "pyvene/axbench-concept16k",
    "concept16k_v2": "pyvene/axbench-concept16k_v2",
}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, repo_id in DATASETS.items():
        print(f"Downloading {repo_id} ...")
        ds = load_dataset(repo_id)

        for split, split_ds in ds.items():
            out_path = os.path.join(DATA_DIR, f"{name}_{split}.json")
            if os.path.exists(out_path):
                print(f"  Skipping {split} split, file already exists: {out_path}")
                continue
            print(f"  Saving {split} split ({len(split_ds)} rows) -> {out_path}")
            split_ds.to_json(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
