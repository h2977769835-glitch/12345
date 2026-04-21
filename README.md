# rec-public-12345

Public runnable snapshot of my generative recommendation work built around a full `slime` checkout plus the `llm4rec` pipeline.

## Repository layout

- `slime/`: full `slime` source tree, including `examples/recommendation_gr/` and the framework entrypoints required to run it
- `llm4rec/`: LLM4Rec-style pipeline with `sasrec`, `rqvae`, and `gr`
- `llm4rec/sasrec/bridge_tencentgr.py`: bridge script that converts `TencentGR-1M` parquet tables into the intermediate files expected by the current `sasrec` pipeline

## What can run directly after clone

### Run the slime recommendation example

The repository includes the full `slime` framework, so the recommendation example keeps its original path:

- `slime/examples/recommendation_gr/rec_data_source.py`
- `slime/examples/recommendation_gr/rec_sft_rollout.py`
- `slime/examples/recommendation_gr/run_rec_sft.sh`

Enter `slime/` and follow the example README there.

### Run the LLM4Rec pipeline

Use the `sasrec -> rqvae -> gr` stages inside `llm4rec/`.

## Data preparation

This repository does not ship the raw TencentGR-1M dataset. Prepare data locally, then use:

- `llm4rec/sasrec/bridge_tencentgr.py`

to generate the intermediate files expected by the current training pipeline.

## Not included

- `TencentGR-1M` raw dataset
- processed bridge outputs
- model checkpoints
- multimodal embedding dumps
- other large generated artifacts
