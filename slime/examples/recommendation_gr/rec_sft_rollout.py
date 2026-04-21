from __future__ import annotations

import logging

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

TOKENIZER = None
SAMPLE_PRINTED = False


def _ensure_tokenizer(args):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        additional_tokens = []
        for tok in ["<|hist_clk_start|>", "<|hist_clk_end|>"]:
            vocab = TOKENIZER.get_vocab()
            if tok not in vocab:
                additional_tokens.append(tok)
        if additional_tokens:
            TOKENIZER.add_special_tokens({"additional_special_tokens": additional_tokens})
            logger.warning(
                "Added missing recommendation special tokens to tokenizer at runtime: %s. "
                "Make sure training model embeddings are resized or tokenizer is saved consistently.",
                additional_tokens,
            )
    return TOKENIZER


def _build_tokens_and_mask(tokenizer, prompt_text: str, label_text: str):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
    token_ids = prompt_ids + label_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(label_ids)
    return token_ids, loss_mask, len(label_ids)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    assert not evaluation, "recommendation_gr minimal example only supports training rollout"

    global SAMPLE_PRINTED
    tokenizer = _ensure_tokenizer(args)
    sample_groups = data_source.get_samples(args.rollout_batch_size)

    for group in sample_groups:
        for sample in group:
            if not isinstance(sample.prompt, str):
                raise TypeError(f"Recommendation prompt must be a string, got {type(sample.prompt)}")
            if not sample.label:
                raise ValueError("Recommendation sample.label is required for SFT rollout")

            token_ids, loss_mask, response_length = _build_tokens_and_mask(tokenizer, sample.prompt, sample.label)

            if len(token_ids) != len(loss_mask):
                raise ValueError(
                    f"Token/mask length mismatch: {len(token_ids)=}, {len(loss_mask)=}"
                )

            sample.tokens = token_ids
            sample.response = sample.label
            sample.response_length = response_length
            sample.loss_mask = loss_mask[-response_length:]
            sample.reward = 0.0
            sample.status = Sample.Status.COMPLETED

            if not SAMPLE_PRINTED:
                logger.info(
                    "recommendation_gr example sample: prompt=%s label=%s response_length=%s total_tokens=%s",
                    sample.prompt,
                    sample.label,
                    response_length,
                    len(token_ids),
                )
                SAMPLE_PRINTED = True

    return RolloutFnTrainOutput(samples=sample_groups)
