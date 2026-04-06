#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a clean Hugging Face model package from a Prime weight export. "
            "Use this when the Prime weight tensors are good but the exported tokenizer/"
            "sidecar files are suspect."
        )
    )
    parser.add_argument("--src", type=Path, required=True, help="Prime weight directory, e.g. outputs/weights/step_50")
    parser.add_argument(
        "--base-tokenizer",
        type=Path,
        required=True,
        help="Base model/tokenizer directory to source tokenizer metadata from",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the rebuilt HF package")
    parser.add_argument("--repo-id", type=str, default=None, help="Optional HF repo id to upload to")
    parser.add_argument("--private", action="store_true", help="Create/upload the HF repo as private")
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="10GB",
        help="Max shard size passed to save_pretrained()",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model/tokenizer",
    )
    return parser.parse_args()


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def main() -> None:
    args = parse_args()
    ensure_exists(args.src, "Prime weight directory")
    ensure_exists(args.base_tokenizer, "Base tokenizer directory")

    args.out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
    )

    model.save_pretrained(args.out, safe_serialization=True, max_shard_size=args.max_shard_size)

    gen_config_path = args.src / "generation_config.json"
    if gen_config_path.exists():
        generation_config = GenerationConfig.from_pretrained(args.src)
        generation_config.save_pretrained(args.out)

    # Copy tokenizer sidecars verbatim from the base tokenizer to avoid
    # tokenizer.save_pretrained() emitting vLLM-incompatible metadata for Qwen3.
    for extra_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
    ):
        src_extra = args.base_tokenizer / extra_name
        if src_extra.exists():
            shutil.copy2(src_extra, args.out / extra_name)

    stable_file = args.src / "STABLE"
    if stable_file.exists():
        shutil.copy2(stable_file, args.out / "STABLE")

    if args.repo_id:
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(args.out),
        )

    print(f"Rebuilt HF package at {args.out}")
    if args.repo_id:
        print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
