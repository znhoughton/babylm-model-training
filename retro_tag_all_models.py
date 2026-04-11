import os
import re
import shutil
import subprocess
from pathlib import Path
import stat
# ==========================================================
#  MODEL CONFIGS (keys only are used here)
# ==========================================================
MODEL_CONFIGS = {
    "znhoughton/opt-babylm-125m-64eps-seed964": {},
    "znhoughton/opt-babylm-350m-64eps-seed964": {},
    "znhoughton/opt-babylm-1.3b-64eps-seed964": {},
    "znhoughton/opt-c4-125m-seed964": {},
    "znhoughton/opt-c4-350m-seed964": {},
    "znhoughton/opt-c4-1.3b-seed964": {},
}

BASE_DIR = Path.cwd() / "tmp_repos"
BASE_DIR.mkdir(exist_ok=True)

# ==========================================================
#  UTILITIES
# ==========================================================


def force_rmtree(path):
    subprocess.run(
        ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
        check=True
    )

def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)

def clone_repo(repo_id, target_dir):
    print(f"\n📥 Cloning {repo_id}")
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    subprocess.run(
        ["git", "clone", f"https://huggingface.co/{repo_id}", str(target_dir)],
        check=True,
        env=env,
    )

def tag_checkpoints(repo_dir):
    print("🏷️  Tagging checkpoints (checkpoint commits only)")

    log = subprocess.check_output(
        ["git", "log", "--pretty=format:%H %s"],
        cwd=repo_dir,
        text=True,
    )

    created = 0
    seen_steps = set()

    for line in log.splitlines():
        sha, msg = line.split(" ", 1)

        # ✅ ONLY tag checkpoint commits
        m = re.search(r"step\s*(\d+).*checkpoint", msg, re.I)
        if not m:
            continue

        step = int(m.group(1))
        if step in seen_steps:
            continue

        tag = f"step-{step}"

        subprocess.run(
            ["git", "tag", "-f", tag, sha],
            cwd=repo_dir,
            check=True,
        )

        seen_steps.add(step)
        created += 1

    print(f"   ✅ Created/updated {created} checkpoint step tags")


def push_tags(repo_dir, batch_size=40):
    print("🚀 Pushing missing tags only")

    local_tags = set(subprocess.check_output(
        ["git", "tag"],
        cwd=repo_dir,
        text=True,
    ).splitlines())

    remote_tags = set(subprocess.check_output(
        ["git", "ls-remote", "--tags", "origin"],
        cwd=repo_dir,
        text=True,
    ).splitlines())

    remote_tag_names = {line.split("refs/tags/")[1] for line in remote_tags if "refs/tags/" in line}

    new_tags = sorted(local_tags - remote_tag_names)

    for i in range(0, len(new_tags), batch_size):
        batch = new_tags[i:i + batch_size]
        print(f"   → Pushing new tags {i + 1}–{i + len(batch)}")

        subprocess.run(
            ["git", "push", "origin", *batch],
            cwd=repo_dir,
            check=True,
        )



# ==========================================================
#  MAIN
# ==========================================================
def main():
    for repo_id in MODEL_CONFIGS:
        repo_name = repo_id.split("/")[-1]
        repo_dir = BASE_DIR / repo_name

        if repo_dir.exists():
            print("🧹 Cleaning up repo directory...")
            force_rmtree(repo_dir)
            print("🧹 Cleanup done")


        try:
            clone_repo(repo_id, repo_dir)
            tag_checkpoints(repo_dir)
            push_tags(repo_dir)
            print("✅ Push returned, moving on...")
            print(f"🎉 Finished {repo_id}")

        except Exception as e:
            print(f"🚨 FAILED for {repo_id}: {e}")

        finally:
            # Always clean up
            if repo_dir.exists():
                print("🧹 Cleaning up repo directory...")
                force_rmtree(repo_dir)
                print("🧹 Cleanup done")


    print("\n🏁 All models processed")

if __name__ == "__main__":
    main()
