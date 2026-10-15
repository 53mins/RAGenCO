# Artifact Manifest

This anonymous supplement contains source code and documentation only.

Not included:

- ML4CO-Bench-101 train, validation, or test datasets.
- Trained RAGenCO checkpoints.
- W&B logs, local Lightning logs, or run metadata.
- Generated Cython C files, compiled extensions, or local build directories.

Expected local artifact layout is chosen by the reviewer or user through command
line flags such as `--training_split`, `--validation_split`, `--test_split`,
`--*_split_label_dir`, `--ckpt_path`, and `--storage_path`.

The optional TSP Cython merge extension can be rebuilt locally with:

```bash
cd diffusion/utils/cython_merge
python setup.py build_ext --inplace
```
