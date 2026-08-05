from typing import List, Optional
from pydantic import BaseModel

class PuzzleDatasetMetadata(BaseModel):
    seq_len: int
    vocab_size: int
    pad_id: int
    # Id of the placeholder token filling the output field. Present for
    # datasets built with the masked-field formulation; absent for legacy
    # (leaky) datasets, which is a useful way to tell them apart.
    mask_id: Optional[int] = None
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]
    
    # Optional fields that might be present in JSON but not strictly required by all consumers
    # Adding for safety if they appear in dataset.json
    # Based on usage in puzzle_dataset.py, only specific fields are accessed.
