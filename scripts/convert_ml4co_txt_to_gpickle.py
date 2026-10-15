#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.readwrite.gpickle import write_gpickle


def convert_linewise(
    input_path: Path,
    out_graph_dir: Path,
    out_label_dir: Path,
    prefix: str,
    start_idx: int = 0,
):
    out_graph_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0
    next_idx = start_idx

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue

            try:
                pos = tokens.index("label")
            except ValueError:
                skipped += 1
                continue

            edge_tokens = tokens[:pos]
            label_tokens = tokens[pos + 1:]
            if len(edge_tokens) % 2 != 0:
                skipped += 1
                continue

            edges = []
            for i in range(0, len(edge_tokens), 2):
                u = int(edge_tokens[i])
                v = int(edge_tokens[i + 1])
                if u != v:
                    edges.append((u, v))

            labels = np.array([int(x) for x in label_tokens], dtype=np.int64)
            if len(edges) == 0:
                skipped += 1
                continue

            n = max(max(u, v) for u, v in edges)
            n = max(n + 1, len(labels))
            if len(labels) < n:
                labels = np.concatenate([labels, np.zeros(n - len(labels), dtype=np.int64)])
            elif len(labels) > n:
                labels = labels[:n]

            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)

            g_path = out_graph_dir / f"{prefix}_{next_idx}.gpickle"
            l_path = out_label_dir / f"{prefix}_{next_idx}_unweighted.result"
            write_gpickle(G, g_path)
            l_path.write_text("\n".join(str(int(x)) for x in labels), encoding="utf-8")
            saved += 1
            next_idx += 1

    return saved, skipped, next_idx


def convert_many_inputs(
    input_paths,
    out_graph_dir: Path,
    out_label_dir: Path,
    prefix: str,
):
    total_saved = 0
    total_skipped = 0
    next_idx = 0

    for input_path in input_paths:
        saved, skipped, next_idx = convert_linewise(
            input_path=Path(input_path),
            out_graph_dir=out_graph_dir,
            out_label_dir=out_label_dir,
            prefix=prefix,
            start_idx=next_idx,
        )
        total_saved += saved
        total_skipped += skipped

    print(
        f"done. saved {total_saved}, skipped {total_skipped}, "
        f"out={out_graph_dir}, {out_label_dir}"
    )


def prepare_train_test_split(
    train_input_glob: str,
    test_input: Path,
    output_root: Path,
    dataset_name: str,
):
    train_inputs = sorted(glob.glob(train_input_glob))
    if not train_inputs:
        raise FileNotFoundError(f"no files matched --train-input-glob={train_input_glob}")
    if not test_input.exists():
        raise FileNotFoundError(f"test file not found: {test_input}")

    train_graph_dir = output_root / f"{dataset_name}_train_gpickle"
    train_label_dir = output_root / f"{dataset_name}_train_label"
    test_graph_dir = output_root / f"{dataset_name}_test_gpickle"
    test_label_dir = output_root / f"{dataset_name}_test_label"

    print(f"[prepare] train inputs: {len(train_inputs)} files")
    print(f"[prepare] test input: {test_input}")

    convert_many_inputs(
        input_paths=train_inputs,
        out_graph_dir=train_graph_dir,
        out_label_dir=train_label_dir,
        prefix=dataset_name,
    )
    convert_many_inputs(
        input_paths=[test_input],
        out_graph_dir=test_graph_dir,
        out_label_dir=test_label_dir,
        prefix=dataset_name,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out-graph-dir", type=Path)
    ap.add_argument("--out-label-dir", type=Path)
    ap.add_argument("--prefix", default="mis_export")
    ap.add_argument("--train-input-glob", type=str)
    ap.add_argument("--test-input", type=Path)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--dataset-name", type=str)
    args = ap.parse_args()

    prepare_mode = (
        args.train_input_glob is not None
        or args.test_input is not None
        or args.output_root is not None
        or args.dataset_name is not None
    )

    if prepare_mode:
        if not all(
            [
                args.train_input_glob is not None,
                args.test_input is not None,
                args.output_root is not None,
                args.dataset_name is not None,
            ]
        ):
            raise ValueError(
                "--train-input-glob, --test-input, --output-root, and --dataset-name "
                "must be provided together in prepare mode."
            )
        prepare_train_test_split(
            train_input_glob=args.train_input_glob,
            test_input=args.test_input,
            output_root=args.output_root,
            dataset_name=args.dataset_name,
        )
        return

    if args.input is None or args.out_graph_dir is None or args.out_label_dir is None:
        raise ValueError(
            "--input, --out-graph-dir, and --out-label-dir are required in single-file mode."
        )

    convert_many_inputs(
        input_paths=[args.input],
        out_graph_dir=args.out_graph_dir,
        out_label_dir=args.out_label_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
