"""
Push all checkpoint-N folders from a local training run to HuggingFace Hub,
one commit per checkpoint, then verify the upload count before signalling
it is safe to delete local files.

Usage:
    python push_checkpoints_to_hub.py \
        --run_dir /dpluth-data/tmp_model/opt-babylm-125m-1ep-dense_964 \
        --hub_model_id znhoughton/opt-babylm-125m-1ep-dense-seed964 \
        --verify_before_delete
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--hub_model_id", required=True)
    p.add_argument("--verify_before_delete", action="store_true")
    return p.parse_args()


def get_local_checkpoints(run_dir: Path) -> list[Path]:
    ckpts = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return ckpts


def count_hub_checkpoints(api: HfApi, repo_id: str) -> int:
    files = list(api.list_repo_files(repo_id))
    dirs = {f.split("/")[0] for f in files if f.split("/")[0].startswith("checkpoint-")}
    return len(dirs)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    repo_id = args.hub_model_id
    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True)

    checkpoints = get_local_checkpoints(run_dir)
    if not checkpoints:
        print(f"No checkpoint-N folders found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(checkpoints)} local checkpoints. Pushing to {repo_id}...")

    for i, ckpt_path in enumerate(checkpoints):
        step = int(ckpt_path.name.split("-")[1])
        commit_msg = f"Training in progress, step {step} checkpoint"
        print(f"  [{i+1}/{len(checkpoints)}] Uploading {ckpt_path.name} ...")
        api.upload_folder(
            folder_path=str(ckpt_path),
            repo_id=repo_id,
            path_in_repo=ckpt_path.name,
            commit_message=commit_msg,
        )

    print(f"\nAll {len(checkpoints)} checkpoints uploaded.")

    if args.verify_before_delete:
        hub_count = count_hub_checkpoints(api, repo_id)
        local_count = len(checkpoints)
        if hub_count == local_count:
            print(f"Verification passed: Hub ({hub_count}) == local ({local_count}).")
            print("Safe to delete local files.")
        else:
            print(
                f"WARNING: Hub count ({hub_count}) != local count ({local_count}). "
                "Do NOT delete local files.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()