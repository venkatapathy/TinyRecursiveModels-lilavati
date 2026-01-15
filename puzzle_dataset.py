import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        
        # Check which datasets have labels_carry (lilavati datasets)
        dataset_paths_with_carry = set()
        for dataset_path in config.dataset_paths:
            for set_name in ["train", "test"]:
                labels_carry_path = os.path.join(dataset_path, set_name, "all__labels_carry.npy")
                if os.path.exists(labels_carry_path):
                    dataset_paths_with_carry.add(dataset_path)
                    break
        
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                # Allow different seq_len when mixing vanilla and lilavati datasets
                # Use the maximum seq_len (lilavati will be longer due to CAR + carry)
                if dataset_path in dataset_paths_with_carry or any(p in dataset_paths_with_carry for p in config.dataset_paths):
                    # When mixing vanilla and lilavati, use max seq_len and max vocab_size
                    # (lilavati has vocab_size=15 with CAR token, vanilla has vocab_size=14)
                    prev_seq_len = max(prev_seq_len, current_metadata.seq_len)
                    prev_vocab_size = max(prev_vocab_size, current_metadata.vocab_size)
                else:
                    assert prev_seq_len == current_metadata.seq_len, f"seq_len mismatch: {prev_seq_len} != {current_metadata.seq_len} (only allowed when mixing vanilla and lilavati datasets)"
                    assert prev_vocab_size == current_metadata.vocab_size, f"vocab_size mismatch: {prev_vocab_size} != {current_metadata.vocab_size} (only allowed when mixing vanilla and lilavati datasets)"
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Detect which dataset paths have labels_carry (lilavati datasets)
        dataset_paths_with_carry = set()
        for dataset_path in self.config.dataset_paths:
            for set_name in self.metadata.sets:
                labels_carry_path = os.path.join(dataset_path, self.split, f"{set_name}__labels_carry.npy")
                if os.path.exists(labels_carry_path):
                    dataset_paths_with_carry.add(dataset_path)
                    break

        # Load data
        self._data = {}
        inputs_carry_data = {}  # Store inputs_carry separately (for lilavati1)
        labels_carry_data = {}  # Store labels_carry separately
        
        # Check if any dataset has labels_carry (lilavati datasets)
        has_any_lilavati = False
        for dataset_path in self.config.dataset_paths:
            for set_name_check in self.metadata.sets:
                labels_carry_path_check = os.path.join(dataset_path, self.split, f"{set_name_check}__labels_carry.npy")
                if os.path.exists(labels_carry_path_check):
                    has_any_lilavati = True
                    break
            if has_any_lilavati:
                break
        
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                
                # For datasets with labels_carry (lilavati), use their inputs
                # For datasets without labels_carry (vanilla), only use their labels
                labels_carry_path = os.path.join(dataset_path, self.split, f"{set_name}__labels_carry.npy")
                inputs_carry_path = os.path.join(dataset_path, self.split, f"{set_name}__inputs_carry.npy")
                has_labels_carry = os.path.exists(labels_carry_path)
                has_inputs_carry = os.path.exists(inputs_carry_path)
                
                if has_labels_carry:
                    # Lilavati dataset: load all fields including inputs
                    self._data[set_name_] = {
                        field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                        for field_name, mmap_mode in field_mmap_modes.items()
                    }
                    # Load labels_carry
                    labels_carry_data[set_name_] = np.load(labels_carry_path, mmap_mode="r")
                    # Load inputs_carry if available (for lilavati1)
                    if has_inputs_carry:
                        inputs_carry_data[set_name_] = np.load(inputs_carry_path, mmap_mode="r")
                else:
                    # Vanilla dataset
                    if has_any_lilavati:
                        # When mixing with lilavati: only load labels (inputs will come from lilavati dataset)
                        # Store labels separately with a special key
                        vanilla_labels_key = f"{set_name_}_vanilla_labels"
                        if not hasattr(self, '_vanilla_labels_data'):
                            self._vanilla_labels_data = {}
                        self._vanilla_labels_data[vanilla_labels_key] = np.load(
                            os.path.join(dataset_path, self.split, f"{set_name}__labels.npy"), mmap_mode="r"
                        )
                    else:
                        # Pure vanilla dataset: load all fields normally
                        self._data[set_name_] = {
                            field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                            for field_name, mmap_mode in field_mmap_modes.items()
                        }
        
        # Store carry data for later use
        if labels_carry_data:
            self._labels_carry_data = labels_carry_data
        else:
            self._labels_carry_data = None
        if inputs_carry_data:
            self._inputs_carry_data = inputs_carry_data
        else:
            self._inputs_carry_data = None


    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
            # Also convert labels_carry if present
            if "labels_carry" in batch:
                batch["labels_carry"][batch["labels_carry"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "inputs_carry": self.metadata.pad_id,  # For inputs_carry padding
                "labels_carry": IGNORE_LABEL_ID,  # For labels_carry padding
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values.get(k, 0)) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch_dict = {
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                }
                # Add carry inputs and labels if available (from lilavati dataset)
                if hasattr(self, '_inputs_carry_data') and self._inputs_carry_data is not None and set_name in self._inputs_carry_data:
                    batch_dict["inputs_carry"] = self._inputs_carry_data[set_name][local_start: local_end]
                if hasattr(self, '_labels_carry_data') and self._labels_carry_data is not None and set_name in self._labels_carry_data:
                    batch_dict["labels_carry"] = self._labels_carry_data[set_name][local_start: local_end]
                
                batch = self._collate_batch(batch_dict)

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_dict = {
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                }
                # Add carry inputs and labels if available (from lilavati dataset)
                if hasattr(self, '_inputs_carry_data') and self._inputs_carry_data is not None and set_name in self._inputs_carry_data:
                    batch_dict["inputs_carry"] = self._inputs_carry_data[set_name][batch_indices]
                if hasattr(self, '_labels_carry_data') and self._labels_carry_data is not None and set_name in self._labels_carry_data:
                    batch_dict["labels_carry"] = self._labels_carry_data[set_name][batch_indices]
                
                batch = self._collate_batch(batch_dict)

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

