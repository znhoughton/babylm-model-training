import argparse
import json
import os
import pathlib

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from tokenizers import models


def mlm_config(
    base_model="roberta-base",
    model_name="roberta-base-bpe",
    vocab=8192,
    hidden_size=256,
    attention=8,
    layers=8,
    intermediate=1024,
    max_len=130,
    bos_id=1,
    pad_id=0,
    eos_id=2,
    unk_id=3,
    mask_id=4,
):
    return AutoConfig.from_pretrained(
        base_model,
        name_or_path=model_name,
        vocab_size=vocab,
        hidden_size=hidden_size,
        num_attention_heads=attention,
        num_hidden_layers=layers,
        intermediate_size=intermediate,
        max_position_embeddings=max_len,
        bos_token_id=bos_id,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        unk_token_id=unk_id,
        mask_token_id=mask_id,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def autoreg_config(
    base_model = "facebook/opt-125m",
    model_name = "opt-125m-bpe",
    vocab=8192,
    hidden_size=256,
    attention=8,
    layers=8,
    intermediate=1024,
    max_len=130,
    bos_id=1,
    pad_id=0,
    eos_id=2,
    unk_id=3,
):
    if "llama" in base_model:
        kv_heads = attention
        config = AutoConfig.from_pretrained(
            base_model,
            name_or_path=model_name,
            vocab_size=vocab,
            hidden_size=hidden_size,
            intermediate_size=intermediate,
            num_attention_heads=attention,
            num_hidden_layers=layers,
            max_position_embeddings=max_len,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            tie_word_embeddings=False,
            num_key_value_heads=kv_heads,
            rope_theta=10000.0,
        )
        config.torch_dtype = None
        return config
    else:
        config = AutoConfig.from_pretrained(
            base_model,
            name_or_path=model_name,
            vocab_size=vocab,
            hidden_size=hidden_size,
            ffn_dim=intermediate,
            num_attention_heads=attention,
            num_hidden_layers=layers,
            max_position_embeddings=max_len,
            word_embed_proj_dim=hidden_size,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            prefix="<s>",
        )
        config.torch_dtype = None
        return config


def main(args):
    model_name = args.model_name
    base_model = args.base_model
    mlm = args.mlm

    # Load existing tokenizer to get special token IDs
    if args.source_tokenizer:
        print(f"Loading tokenizer from {args.source_tokenizer}")
        source_tok = AutoTokenizer.from_pretrained(args.source_tokenizer)
        
        # Get special token IDs from the source tokenizer
        bos_id = source_tok.bos_token_id if source_tok.bos_token_id is not None else 1
        pad_id = source_tok.pad_token_id if source_tok.pad_token_id is not None else 0
        eos_id = source_tok.eos_token_id if source_tok.eos_token_id is not None else 2
        unk_id = source_tok.unk_token_id if source_tok.unk_token_id is not None else 3
        vocab = len(source_tok)
        
        print(f"Using vocab size: {vocab}")
        print(f"Special token IDs - BOS: {bos_id}, PAD: {pad_id}, EOS: {eos_id}, UNK: {unk_id}")
    else:
        # Default IDs for BPE tokenizer
        bos_id = 1
        pad_id = 0
        eos_id = 2
        unk_id = 3
        vocab = args.vocab
        print(f"Using default special token IDs and vocab size: {vocab}")

    # Create config
    if mlm:
        cfg = mlm_config(
            base_model,
            model_name,
            vocab,
            args.hidden_size,
            args.attention_heads,
            args.layers,
            args.intermediate_size,
            args.max_len,
            bos_id,
            pad_id,
            eos_id,
            unk_id,
            4,  # mask_id
        )
    else:
        cfg = autoreg_config(
            base_model,
            model_name,
            vocab,
            args.hidden_size,
            args.attention_heads,
            args.layers,
            args.intermediate_size,
            args.max_len,
            bos_id,
            pad_id,
            eos_id,
            unk_id,
        )

    # Save config
    pathlib.Path(f"models/{model_name}").mkdir(parents=True, exist_ok=True)
    cfg._name_or_path = model_name
    cfg.save_pretrained(f"models/{model_name}")
    
    print(f"✓ Saved config to models/{model_name}")
    print(f"  Config: {cfg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "-b", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_name", "-m", type=str, default="smolm-mlm")
    parser.add_argument("--source_tokenizer", "-s", type=str, default=None,
                       help="Path to existing tokenizer to copy vocab size and special tokens from")
    parser.add_argument("--train_file", "-t", type=str)
    parser.add_argument("--dataset_config", "-dc", type=str, default=None)
    parser.add_argument("--mlm", action="store_true")
    parser.add_argument("--bpe", action="store_true")
    parser.add_argument("--word", action="store_true")
    parser.add_argument("--vocab", "-v", type=int, default=8192)
    parser.add_argument("--hidden_size", "-hs", type=int, default=256)
    parser.add_argument("--intermediate_size", "-i", type=int, default=1024)
    parser.add_argument("--max_len", "-l", type=int, default=128)
    parser.add_argument("--layers", "-y", type=int, default=8)
    parser.add_argument("--attention_heads", "-a", type=int, default=8)
    parser.add_argument("--from_iterator", "-f", action="store_true")
    args = parser.parse_args()

    print(args)

    main(args)