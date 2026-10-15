"""ATSP (Asymmetric Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch
from torch_geometric.data import Data as GraphData

class ATSPGraphDataset(torch.utils.data.Dataset):
    """Simple dataset for ATSP instances stored as lines:
    "<flattened_cost_matrix_values> output <tour_indices>"

    - cost matrix is flattened row-major (N*N floats)
    - tour is 1-based indices; can be length N or N+1 (closed)
    """

    def __init__(self, data_file, graph_type="directed"):
        self.data_file = data_file
        self.file_lines = open(data_file).read().splitlines()
        self.graph_type = graph_type
        print(f'Loaded ATSP "{data_file}" with {len(self.file_lines)} lines')

    def __len__(self):
        return len(self.file_lines)

    def _parse_line(self, line):
        # Expect a single ' output ' separator (with surrounding spaces)
        parts = line.strip().split(" output ", 1)
        if len(parts) != 2:
            raise ValueError("Each line must contain ' output ' separator")
        vals_str, tour_str = parts

        vals = [float(x) for x in vals_str.split()]
        tour_tokens = [int(x) for x in tour_str.split()]

        # infer N from flattened cost length
        n_vals = len(vals)
        N = int(np.sqrt(n_vals))
        if N * N != n_vals:
            raise ValueError(f"Flattened cost length {n_vals} is not a perfect square")

        cost_matrix = np.array(vals, dtype=np.float32).reshape(N, N)

        # convert 1-based tour to 0-based and ensure closed tour
        tour = np.array(tour_tokens, dtype=np.int64) - 1
        if tour.shape[0] == N:
            tour = np.concatenate([tour, tour[:1]])
        elif tour.shape[0] != N + 1:
            raise ValueError(f"Tour length invalid: {tour.shape[0]} (expected {N+1} or {N})")

        # build directed adjacency (optionally symmetrize)
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(tour.shape[0] - 1):
            a = int(tour[i])
            b = int(tour[i + 1])
            adj[a, b] = 1.0
            if self.graph_type != "directed":
                adj[b, a] = 1.0

        return cost_matrix, adj, tour

    def __getitem__(self, idx):
        line = self.file_lines[idx]
        cost_matrix, adj_matrix, tour = self._parse_line(line)

        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            torch.from_numpy(cost_matrix).float(),    # [N, N]
            torch.from_numpy(adj_matrix).float(),     # [N, N]
            torch.from_numpy(tour).long(),            # [N+1]
        )