# recommendation_gr

This example ports `马上转码（LLM4Rec）/gr` style generative recommendation SFT into `slime`.

## Goal

Train an LLM to generate the target Semantic ID sequence from a user's historical Semantic ID sequence.

## Expected data files

### 1. `train_data.json`

```json
{
  "user_1": ["itemA", "itemB", "itemC"],
  "user_2": ["itemX", "itemY"]
}
```

### 2. `item2token.txt`

Tab-separated mapping from item id to semantic token string:

```text
itemA	<id_1><id_42><id_7>
itemB	<id_9><id_12><id_3>
```

## Environment variables

- `REC_TRAIN_JSON`: path to `train_data.json`
- `REC_ITEM2TOKEN_PATH`: path to `item2token.txt`
- `HF_MODEL_PATH`: base Hugging Face model path
- `TORCH_DIST_CKPT`: Megatron torch_dist checkpoint path
- `REC_SAVE_PATH`: output checkpoint path

## Files

- `rec_data_source.py`: recommendation dataset loader for slime
- `rec_sft_rollout.py`: reproduces `gr/custom_dataset.py` style tokenization and loss masking
- `run_rec_sft.sh`: minimal SFT launch script
- `debug_rec_data.json`: tiny debug dataset

## Notes

This is the minimal SFT migration only. It does not include recommendation reward / RL yet.
