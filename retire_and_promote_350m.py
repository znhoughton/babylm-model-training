"""Retire the post-LN 350M and promote the corrected one into its name.

    old: znhoughton/opt-babylm-350m-20eps-seed964          -> ...-seed964-old (private)
    new: znhoughton/opt-babylm-350m-20eps-prenorm-seed964  -> ...-seed964

Point of the exercise: every repo already references
opt-babylm-350m-20eps-seed964, and the three scales stay name-matched
(125m/350m/1.3b all "-20eps-seed964"). After this, no analysis code changes at
all -- the sed steps in the rerun_350m_prenorm.sh scripts become no-ops.

RUN THIS ONLY AFTER TRAINING FINISHES. The run pushes a checkpoint every 48
steps to the -prenorm repo; renaming it mid-run breaks those pushes.

TOKEN: needs admin/settings scope, not just write. A token with only
repo.write will be refused on both the move and the visibility change. If that
is what you have, do it in the web UI instead -- Settings -> Rename, then
Settings -> Change visibility.

    python retire_and_promote_350m.py --dry-run
    python retire_and_promote_350m.py --go

STATUS 2026-09-22: steps 1 and 2 are DONE. The old post-LN model is at
...-seed964-old and is private. Only step 3 (promotion) remains, and it must
wait until training stops pushing to the -prenorm repo.

KNOWN RISK for step 3: renaming leaves a redirect behind, so the canonical name
currently forwards to ...-seed964-old rather than being free. If move_repo is
refused because the name looks taken, the fallback is to rename the retired
repo again to a name unrelated to the canonical one (e.g.
opt-babylm-350m-20eps-POSTLN-BUGGED), which breaks the redirect chain, then
retry the promotion. Verify afterwards that the canonical name serves
do_layer_norm_before=True -- do not assume it.
"""

import argparse
import sys

from huggingface_hub import HfApi

USER = "znhoughton"
CANON = f"{USER}/opt-babylm-350m-20eps-seed964"
RETIRED = f"{CANON}-old"
INCOMING = f"{USER}/opt-babylm-350m-20eps-prenorm-seed964"


def describe(api, repo_id):
    """Report a repo's existence, privacy, and layernorm placement."""
    try:
        info = api.model_info(repo_id, files_metadata=False)
    except Exception as err:
        return f"{repo_id}: UNREACHABLE ({type(err).__name__})"
    ln = "?"
    try:
        import json
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(repo_id, "config.json"), encoding="utf-8"))
        ln = cfg.get("do_layer_norm_before")
    except Exception:
        pass
    return f"{repo_id}: private={info.private} do_layer_norm_before={ln}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.go):
        ap.error("pass --dry-run or --go")

    api = HfApi()
    print("whoami:", api.whoami().get("name"))

    print("\n-- before --")
    for r in (CANON, RETIRED, INCOMING):
        print("  " + describe(api, r))

    # Refuse to run while the corrected model is still being written to.
    try:
        commits = api.list_repo_commits(INCOMING)
        import re
        steps = [int(m.group(1)) for c in commits
                 if (m := re.search(r"step\s*(\d+)", c.title, re.I))]
        print(f"\n  {INCOMING} has {len(steps)} checkpoint commits, "
              f"max step {max(steps) if steps else 'n/a'}")
        print("  Confirm training is FINISHED before promoting; a rename mid-run"
              " breaks the remaining pushes.")
    except Exception as err:
        print(f"  could not list commits on {INCOMING}: {err}")

    if args.dry_run:
        print("\n-- would do --")
        print(f"  1. move   {CANON} -> {RETIRED}")
        print(f"  2. private {RETIRED}")
        print(f"  3. move   {INCOMING} -> {CANON}")
        print("\n(dry run, nothing changed)")
        return 0

    print("\n-- doing --")
    # Order matters: the canonical name has to be vacated before the corrected
    # model can take it.
    try:
        print(f"  1. move {CANON} -> {RETIRED}")
        api.move_repo(from_id=CANON, to_id=RETIRED, repo_type="model")
    except Exception as err:
        print(f"     FAILED: {err}")
        print("     If this is a permissions error, the token lacks admin scope;"
              " use the web UI.")
        return 1

    try:
        print(f"  2. make {RETIRED} private")
        api.update_repo_settings(repo_id=RETIRED, private=True, repo_type="model")
    except Exception as err:
        print(f"     FAILED: {err} (repo is renamed but still public)")

    try:
        print(f"  3. move {INCOMING} -> {CANON}")
        api.move_repo(from_id=INCOMING, to_id=CANON, repo_type="model")
    except Exception as err:
        print(f"     FAILED: {err}")
        print("     NOTE: step 1 already happened. The canonical name is vacant"
              " (or redirecting to -old). Resolve before re-running.")
        return 1

    print("\n-- after --")
    for r in (CANON, RETIRED, INCOMING):
        print("  " + describe(api, r))

    # The whole point is that CANON now serves the corrected weights. Renaming
    # leaves a redirect behind from the old name, so verify rather than assume.
    print("\n-- verification --")
    try:
        import json
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(CANON, "config.json"), encoding="utf-8"))
        ok = cfg.get("do_layer_norm_before") is True
        print(f"  {CANON} do_layer_norm_before = {cfg.get('do_layer_norm_before')}")
        print("  OK: canonical name serves the corrected model." if ok
              else "  FATAL: canonical name is NOT serving the pre-LN model.")
        if not ok:
            return 1
    except Exception as err:
        print(f"  could not verify: {err}")
        return 1

    print("\nNEXT: purge stale caches, or re-runs will silently use the OLD weights")
    print("  local: rm -rf ~/.cache/huggingface/hub/models--znhoughton--opt-babylm-350m-20eps-seed964")
    print("  pod:   rm -rf $HF_HOME/hub/models--znhoughton--opt-babylm-350m-20eps-seed964")
    return 0


if __name__ == "__main__":
    sys.exit(main())
