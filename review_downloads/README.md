# Large Assets for Anonymous Review

The mirror has an 8 MB per-file limit. These binary parts are each at most 7,000,000 bytes. They preserve the original bytes of the LoRA adapter, tokenizer, method SVG and video-case PDF. They are stored as ordinary Git blobs, not LFS pointers.

1. Download the repository ZIP from the anonymous mirror and extract it. Alternatively download all files in this directory, preserving their names, together with `scripts/restore_review_assets.py`.
2. Run `python scripts/restore_review_assets.py` from the extracted repository.
3. Run `python scripts/verify_equiaffect.py` to check the restored LoRA adapter.

The restore script checks each part and the final file against `manifest.json`, then restores the original relative paths. It uses only Python's standard library. `--verify-only` checks the parts without writing files. Use `scripts/download_base_model.py` to obtain the separately licensed base model from its official source.
