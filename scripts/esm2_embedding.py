import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

# Available ESM2 models by size:
# "facebook/esm2_t6_8M_UR50D"    — 8M params, fastest
# "facebook/esm2_t12_35M_UR50D"  — 35M params
# "facebook/esm2_t30_150M_UR50D" — 150M params
# "facebook/esm2_t33_650M_UR50D" — 650M params, best quality
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


def load_model(model_name: str = MODEL_NAME, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def embed_sequences(
    sequences: list[str],
    tokenizer,
    model,
    device: str,
    pool: str = "mean",
) -> torch.Tensor:
    """
    Embed a list of protein sequences using ESM2.

    Args:
        sequences: List of amino acid sequences (uppercase single-letter codes).
        tokenizer:  ESM2 tokenizer.
        model:      ESM2 model.
        device:     "cpu" or "cuda".
        pool:       How to aggregate per-token embeddings into one vector.
                    "mean"  — average over non-special tokens (recommended)
                    "cls"   — use the [CLS] token representation

    Returns:
        Tensor of shape (N, hidden_size).
    """
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state  # (N, seq_len, hidden_size)

    if pool == "cls":
        return hidden[:, 0, :]

    # Mean-pool over real (non-padding) token positions, excluding [CLS] and [EOS]
    attention_mask = inputs["attention_mask"]  # (N, seq_len)
    # Zero out [CLS] (position 0) and [EOS] (last non-pad position) if desired,
    # but averaging the full attended region is standard and works well.
    mask = attention_mask.unsqueeze(-1).float()  # (N, seq_len, 1)
    summed = (hidden * mask).sum(dim=1)           # (N, hidden_size)
    counts = mask.sum(dim=1)                      # (N, 1)
    return summed / counts


def load_records_from_json(path: str | Path) -> list[dict]:
    """
    Load records from a benchmark dataset JSON file.

    Expected format:
        [
          {"ID": "2E8D_A", "LABEL": "AMYLOID", "Sequence": "SNFLNCY...", ...},
          ...
        ]

    Args:
        path: Path to the JSON file.

    Returns:
        List of record dicts, each guaranteed to have a "Sequence" key.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at the top level, got {type(data).__name__}.")
    if not all(isinstance(r, dict) for r in data):
        raise ValueError("Every element in the JSON array must be an object with a 'Sequence' field.")
    missing = [i for i, r in enumerate(data) if "Sequence" not in r]
    if missing:
        raise ValueError(f"Records at indices {missing} are missing the 'Sequence' field.")

    print(f"Loaded {len(data)} record(s) from {path}")
    return data


def load_sequences_from_json(path: str | Path) -> tuple[list[str], list[str]]:
    """
    Load sequences and IDs from a benchmark dataset JSON file.

    Expected format:
        [{"ID": "2E8D_A", "Sequence": "SNFLNCY...", ...}, ...]

    Returns:
        ids       — list of record IDs (empty string if "ID" key absent)
        sequences — list of amino acid sequence strings
    """
    records = load_records_from_json(path)
    ids = [r.get("ID", "") for r in records]
    sequences = [r["Sequence"] for r in records]
    return ids, sequences


def main():
    parser = argparse.ArgumentParser(description="Embed protein sequences with ESM2.")
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        help='Benchmark JSON file: [{"ID": "...", "Sequence": "...", ...}, ...]',
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        default="embeddings.pt",
        help="Output file for saved embeddings (default: embeddings.pt)",
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL_NAME,
        help=f"ESM2 model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--pool",
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy (default: mean)",
    )
    args = parser.parse_args()

    if args.input:
        ids, sequences = load_sequences_from_json(args.input)
    else:
        print("No --input file given, using built-in demo sequences.\n")
        ids = ["demo_1", "demo_2", "demo_3"]
        sequences = [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
            "ACDEFGHIKLMNPQRSTVWY",
            "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPNEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV",
        ]

    print(f"Loading model: {args.model}")
    tokenizer, model, device = load_model(args.model)
    print(f"Running on: {device}")
    print(f"Hidden size: {model.config.hidden_size}\n")

    embeddings = embed_sequences(sequences, tokenizer, model, device, pool=args.pool)

    for id_, seq, emb in zip(ids, sequences, embeddings):
        print(f"ID: {id_}  length={len(seq)}")
        print(f"  Embedding shape : {tuple(emb.shape)}")
        print(f"  Embedding norm  : {emb.norm().item():.4f}")
        print(f"  First 5 values  : {emb[:5].tolist()}\n")

    torch.save({"ids": ids, "sequences": sequences, "embeddings": embeddings}, args.output)
    print(f"Saved embeddings to {args.output}")
    return embeddings


if __name__ == "__main__":
    main()
