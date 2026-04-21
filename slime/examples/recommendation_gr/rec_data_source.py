from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import torch

from slime.utils.misc import load_function
from slime.utils.types import Sample


class RecommendationDataSource:
    """Minimal data source for recommendation SFT.

    Input file format:
    {
        "user_1": ["itemA", "itemB", "itemC"],
        ...
    }

    item2token file format (tab separated):
        item_id\t<id_1><id_2><id_3>
    """

    def __init__(self, args):
        self.args = args
        self.prompt_data = Path(args.prompt_data)
        self.item2token_path = Path(getattr(args, "rec_item2token_path", "") or os.environ["REC_ITEM2TOKEN_PATH"])
        self.max_seq_length = int(getattr(args, "rec_max_seq_length", 0) or os.environ.get("REC_MAX_SEQ_LENGTH", 150))
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.epoch_id = 0
        self.buffer = []

        if getattr(args, "buffer_filter_path", None) is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(args.buffer_filter_path)

        self.item2token = self._load_item2token(self.item2token_path)
        self.samples = self._build_samples(self.prompt_data)

        if getattr(self.args, "rollout_shuffle", False):
            self._shuffle(self.epoch_id)

    @staticmethod
    def _load_item2token(path: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                item_id, token = line.split("\t", 1)
                mapping[item_id] = token
        return mapping

    def _build_samples(self, path: Path) -> list[Sample]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples: list[Sample] = []
        for user_id, item_sequence in data.items():
            if not isinstance(item_sequence, list) or len(item_sequence) < 2:
                continue

            target_item = None
            for item in reversed(item_sequence):
                key = str(item)
                if key in self.item2token:
                    target_item = key
                    break

            if target_item is None:
                continue

            history_tokens: list[str] = []
            for item in item_sequence:
                key = str(item)
                if key == target_item:
                    continue
                if key in self.item2token:
                    history_tokens.append(self.item2token[key])

            history_tokens = history_tokens[-self.max_seq_length :]
            if not history_tokens:
                continue

            prompt = "<|hist_clk_start|>" + "".join(history_tokens) + "<|hist_clk_end|>"
            label = self.item2token[target_item]

            sample = Sample(
                prompt=prompt,
                label=label,
                metadata={
                    "user_id": str(user_id),
                    "target_item": target_item,
                    "history_token_count": len(history_tokens),
                },
            )
            samples.append(sample)

        if not samples:
            raise ValueError(
                f"No valid recommendation samples were built from {path}. "
                "Check train_data.json and item2token mapping."
            )
        return samples

    def _shuffle(self, epoch_id: int):
        rng = random.Random(getattr(self.args, "rollout_seed", 42) + epoch_id)
        rng.shuffle(self.samples)

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []
        return self.buffer_filter(self.args, None, self.buffer, num_samples)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = self._get_samples_from_buffer(num_samples)
        num_samples -= len(groups)
        if num_samples <= 0:
            return groups

        prompt_samples: list[Sample] = []
        if self.sample_offset + num_samples <= len(self.samples):
            prompt_samples = self.samples[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            prompt_samples = self.samples[self.sample_offset :]
            num_samples -= len(prompt_samples)
            self.epoch_id += 1
            if getattr(self.args, "rollout_shuffle", False):
                self._shuffle(self.epoch_id)
            prompt_samples += self.samples[:num_samples]
            self.sample_offset = num_samples

        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples: list[list[Sample]]):
        if not samples:
            return
        for group in samples:
            if len(group) != self.args.n_samples_per_prompt:
                raise ValueError(
                    f"Each sample group must have {self.args.n_samples_per_prompt} samples, got {len(group)}"
                )
            self.buffer.append(group)

    def save(self, rollout_id):
        if getattr(self.args, "save", None) is None:
            return
        state = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
        }
        path = Path(self.args.save) / "rollout" / f"rec_data_source_state_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, rollout_id=None):
        if getattr(self.args, "load", None) is None or rollout_id is None:
            return
        path = Path(self.args.load) / "rollout" / f"rec_data_source_state_{rollout_id}.pt"
        if not path.exists():
            return
        state = torch.load(path)
        self.sample_offset = state.get("sample_offset", 0)
        self.epoch_id = state.get("epoch_id", 0)
        self.sample_group_index = state.get("sample_group_index", 0)
        self.sample_index = state.get("sample_index", 0)
        if getattr(self.args, "rollout_shuffle", False):
            self._shuffle(self.epoch_id)

    def __len__(self) -> int:
        return len(self.samples) + len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
