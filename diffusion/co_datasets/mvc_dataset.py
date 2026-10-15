"""MVC dataset.

The on-disk layout matches MIS:
  - one graph per ``.gpickle``
  - one ``*_unweighted.result`` label file per graph

The stored labels remain the original MVC cover indicators. The RAGenCO MVC model
internally converts them into complement independent-set supervision so it can
reuse the MIS backbone and decoding pipeline.
"""

from diffusion.co_datasets.mis_dataset import MISDataset


class MVCDataset(MISDataset):
    """Alias of MISDataset with MVC-specific semantics."""

