import torch
import math
import csv
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODELS = [
    ("OPT-125M", "znhoughton/opt-babylm-125m-20eps-seed964"),
    ("OPT-350M", "znhoughton/opt-babylm-350m-20eps-seed964"),
    ("OPT-1.3B",  "znhoughton/opt-babylm-1.3b-20eps-seed964"),
]
DATASET   = "znhoughton/babylm-150m-v3"
BLOCK_SIZE  = 1024
BATCH_SIZE  = 4
MAX_CHUNKS  = 500   # ~500k tokens — enough for stable perplexity estimate
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset(DATASET, trust_remote_code=True)
print(ds)

if "validation" in ds:
    val_raw = ds["validation"]
elif "dev" in ds:
    val_raw = ds["dev"]
else:
    n = len(ds["train"])
    val_raw = ds["train"].select(range(int(0.05 * n)))

print(f"Validation examples: {len(val_raw)}")

# Detect text column
text_col = "text" if "text" in val_raw.column_names else val_raw.column_names[0]
print(f"Text column: '{text_col}'")

results = []

for model_name, model_id in MODELS:
    print(f"\n{'='*60}\nLoading {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Concatenate and chunk validation text
    all_ids = []
    for ex in val_raw:
        all_ids.extend(tokenizer.encode(ex[text_col]))
        if len(all_ids) >= MAX_CHUNKS * BLOCK_SIZE:
            break

    chunks = [
        all_ids[i : i + BLOCK_SIZE]
        for i in range(0, len(all_ids) - BLOCK_SIZE, BLOCK_SIZE)
    ][:MAX_CHUNKS]

    print(f"Chunks to evaluate: {len(chunks)}")

    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            ids   = torch.tensor(batch, dtype=torch.long).to(DEVICE)
            loss  = model(ids, labels=ids).loss.item()
            total_loss += loss
            n_batches  += 1
            if n_batches % 25 == 0:
                running_ppl = math.exp(total_loss / n_batches)
                print(f"  batch {n_batches}/{math.ceil(len(chunks)/BATCH_SIZE)}  "
                      f"loss={total_loss/n_batches:.4f}  ppl={running_ppl:.2f}")

    avg_loss = total_loss / n_batches
    ppl      = math.exp(avg_loss)
    print(f">> {model_name}: loss={avg_loss:.4f}, perplexity={ppl:.2f}")
    results.append({"model": model_name, "val_loss": round(avg_loss, 4),
                    "perplexity": round(ppl, 2)})

    del model
    torch.cuda.empty_cache()

# ── Save CSV ───────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "babylm_val_results.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "val_loss", "perplexity"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to {out}")
for r in results:
    print(r)
