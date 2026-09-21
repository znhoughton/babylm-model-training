"""Fail loudly if a generated model config is not what the run intended.

tokenizer_and_config.py builds configs with AutoConfig.from_pretrained(base),
which inherits every field it does not explicitly override. facebook/opt-350m
is the one OPT size Meta shipped with a post-LN block
(do_layer_norm_before=False) and a 512-dim embedding projection, so a 350m-based
config silently produced a model that differed from the other sizes in
architecture rather than only in scale.

Run this after building a config and before spending GPU hours on it:

    python check_config.py models/opt-babylm-350m-20eps \
        --hidden 1024 --layers 24 --heads 16 --ffn 4096
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="directory holding config.json")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--ffn", type=int, required=True)
    args = parser.parse_args()

    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    problems = []

    # The bug this file exists for.
    if config.get("do_layer_norm_before") is not True:
        problems.append(
            f"do_layer_norm_before={config.get('do_layer_norm_before')!r} "
            "(post-LN). Every size must be pre-LN to be comparable."
        )

    # The other opt-350m quirk, already overridden but worth pinning down.
    if config.get("word_embed_proj_dim") != config.get("hidden_size"):
        problems.append(
            f"word_embed_proj_dim={config.get('word_embed_proj_dim')} "
            f"!= hidden_size={config.get('hidden_size')}"
        )

    expected = {
        "hidden_size": args.hidden,
        "num_hidden_layers": args.layers,
        "num_attention_heads": args.heads,
        "ffn_dim": args.ffn,
    }
    for key, want in expected.items():
        if config.get(key) != want:
            problems.append(f"{key}={config.get(key)}, expected {want}")

    if problems:
        print(f"CONFIG CHECK FAILED for {config_path}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"config OK: {args.model_path} "
        f"(pre-LN, hidden={args.hidden}, layers={args.layers}, "
        f"heads={args.heads}, ffn={args.ffn})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
