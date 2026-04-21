from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

ITEM_FEAT_IDS = ["100", "101", "102", "112", "114", "115", "116", "117", "118", "119", "120", "121", "122"]
USER_FEAT_IDS = ["103", "104", "105", "106", "107", "108", "109", "110"]
MM_EMB_SPECS = {
    "81": ("emb_81_32_parquet", 32),
    "82": ("emb_82_1024_parquet", 1024),
    "83": ("emb_83_3584_parquet", 3584),
    "84": ("emb_84_4096_parquet", 4096),
    "85": ("emb_85_3584_parquet", 3584),
    "86": ("emb_86_3584_parquet", 3584),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge TencentGR-1M parquet data to LLM4Rec sasrec inputs")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to TencentGR-1M root directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for processed sasrec inputs")
    parser.add_argument(
        "--indexer-path",
        type=Path,
        default=None,
        help="Optional path to indexer.pkl. Defaults to <dataset-root>/indexer.pkl if it exists.",
    )
    parser.add_argument(
        "--mm-emb-ids",
        nargs="+",
        default=["81", "82"],
        choices=sorted(MM_EMB_SPECS.keys()),
        help="Which multimodal embedding groups to export",
    )
    parser.add_argument(
        "--skip-mm-emb",
        action="store_true",
        help="Skip exporting creative_emb files",
    )
    parser.add_argument(
        "--skip-gr-train-data",
        action="store_true",
        help="Skip generating gr/train_data.json convenience file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Arrow batch size while scanning parquet files",
    )
    return parser.parse_args()


def list_parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    return files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, list):
        return [normalize_scalar(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_scalar(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def load_indexer(indexer_path: Path | None) -> dict[str, Any] | None:
    if indexer_path is None or not indexer_path.exists():
        return None
    with indexer_path.open("rb") as f:
        return pickle.load(f)


def build_item_feat_dict(dataset_root: Path, batch_size: int) -> dict[str, dict[str, Any]]:
    item_feat_dir = dataset_root / "item_feat"
    item_feat_dict: dict[str, dict[str, Any]] = {}
    for parquet_path in tqdm(list_parquet_files(item_feat_dir), desc="item_feat parquet"):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                item_id = row["item_id"]
                feat_dict: dict[str, Any] = {}
                for feat_id in ITEM_FEAT_IDS:
                    value = normalize_scalar(row.get(feat_id))
                    if value is not None:
                        feat_dict[feat_id] = value
                item_feat_dict[str(item_id)] = feat_dict
    return item_feat_dict


def build_user_feat_dict(dataset_root: Path, batch_size: int) -> dict[str, dict[str, Any]]:
    user_feat_dir = dataset_root / "user_feat"
    user_feat_dict: dict[str, dict[str, Any]] = {}
    for parquet_path in tqdm(list_parquet_files(user_feat_dir), desc="user_feat parquet"):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                user_id = row["user_id"]
                feat_dict: dict[str, Any] = {}
                for feat_id in USER_FEAT_IDS:
                    value = normalize_scalar(row.get(feat_id))
                    if value is not None:
                        feat_dict[feat_id] = value
                user_feat_dict[str(user_id)] = feat_dict
    return user_feat_dict


def build_seq_files(
    dataset_root: Path,
    output_root: Path,
    user_feat_dict: dict[str, dict[str, Any]],
    item_feat_dict: dict[str, dict[str, Any]],
    batch_size: int,
    emit_gr_train_data: bool,
) -> dict[str, Any]:
    seq_dir = dataset_root / "seq"
    seq_jsonl_path = output_root / "seq.jsonl"
    seq_offsets_path = output_root / "seq_offsets.pkl"
    gr_dir = output_root / "gr"
    gr_train_data: dict[str, list[int]] = {}
    offsets: list[int] = []
    user_ids_in_order: list[int] = []
    written_users = 0
    written_events = 0

    if emit_gr_train_data:
        ensure_dir(gr_dir)

    with seq_jsonl_path.open("wb") as fout:
        for parquet_path in tqdm(list_parquet_files(seq_dir), desc="seq parquet"):
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    user_id = int(row["user_id"])
                    events = row.get("seq") or []
                    if not events:
                        continue

                    events = sorted(events, key=lambda x: (x.get("timestamp") is None, x.get("timestamp", 0)))
                    user_feat = user_feat_dict.get(str(user_id), {})
                    output_records: list[list[Any]] = []
                    gr_items: list[int] = []

                    for event in events:
                        item_id = event.get("item_id")
                        if item_id is None:
                            continue
                        item_id = int(item_id)
                        item_feat = item_feat_dict.get(str(item_id), {})
                        action_type = normalize_scalar(event.get("action_type"))
                        timestamp = normalize_scalar(event.get("timestamp"))
                        output_records.append([user_id, item_id, user_feat, item_feat, action_type, timestamp])
                        gr_items.append(item_id)

                    if len(output_records) < 2:
                        continue

                    offsets.append(fout.tell())
                    fout.write(json.dumps(output_records, ensure_ascii=False).encode("utf-8"))
                    fout.write(b"\n")
                    user_ids_in_order.append(user_id)
                    written_users += 1
                    written_events += len(output_records)

                    if emit_gr_train_data:
                        gr_train_data[str(user_id)] = gr_items

    with seq_offsets_path.open("wb") as f:
        pickle.dump(offsets, f)

    if emit_gr_train_data:
        with (gr_dir / "train_data.json").open("w", encoding="utf-8") as f:
            json.dump(gr_train_data, f, ensure_ascii=False)

    with (output_root / "seq_user_ids.json").open("w", encoding="utf-8") as f:
        json.dump(user_ids_in_order, f, ensure_ascii=False)

    return {
        "written_users": written_users,
        "written_events": written_events,
        "seq_jsonl_path": str(seq_jsonl_path),
        "seq_offsets_path": str(seq_offsets_path),
    }


def export_mm_embeddings(dataset_root: Path, output_root: Path, feat_ids: list[str], batch_size: int) -> dict[str, Any]:
    mm_root = dataset_root / "mm_emb"
    creative_root = output_root / "creative_emb"
    ensure_dir(creative_root)
    stats: dict[str, Any] = {}

    for feat_id in tqdm(feat_ids, desc="mm_emb groups"):
        subdir_name, dim = MM_EMB_SPECS[feat_id]
        parquet_dir = mm_root / subdir_name
        out_dir = creative_root / f"emb_{feat_id}_{dim}"
        ensure_dir(out_dir)
        out_path = out_dir / "part-00000.json"
        record_count = 0

        with out_path.open("w", encoding="utf-8") as fout:
            for parquet_path in tqdm(list_parquet_files(parquet_dir), desc=f"mm_emb #{feat_id}", leave=False):
                parquet_file = pq.ParquetFile(parquet_path)
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["anonymous_cid", "emb"]):
                    for row in batch.to_pylist():
                        anonymous_cid = row.get("anonymous_cid")
                        emb = row.get("emb")
                        if anonymous_cid is None or emb is None:
                            continue
                        payload = {"anonymous_cid": str(anonymous_cid), "emb": [float(v) for v in emb]}
                        fout.write(json.dumps(payload, ensure_ascii=False))
                        fout.write("\n")
                        record_count += 1

        stats[feat_id] = {"records": record_count, "output": str(out_path)}

    return stats


def copy_indexer(indexer_path: Path | None, output_root: Path) -> bool:
    if indexer_path is None or not indexer_path.exists():
        return False
    with indexer_path.open("rb") as src, (output_root / "indexer.pkl").open("wb") as dst:
        dst.write(src.read())
    return True


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    indexer_path = args.indexer_path
    if indexer_path is None:
        default_indexer = dataset_root / "indexer.pkl"
        if default_indexer.exists():
            indexer_path = default_indexer

    indexer = load_indexer(indexer_path)
    copied_indexer = copy_indexer(indexer_path, output_root)

    item_feat_dict = build_item_feat_dict(dataset_root, args.batch_size)
    write_json(output_root / "item_feat_dict.json", item_feat_dict)

    user_feat_dict = build_user_feat_dict(dataset_root, args.batch_size)
    write_json(output_root / "user_feat_dict.json", user_feat_dict)

    seq_stats = build_seq_files(
        dataset_root=dataset_root,
        output_root=output_root,
        user_feat_dict=user_feat_dict,
        item_feat_dict=item_feat_dict,
        batch_size=args.batch_size,
        emit_gr_train_data=not args.skip_gr_train_data,
    )

    mm_stats: dict[str, Any] = {}
    if not args.skip_mm_emb:
        mm_stats = export_mm_embeddings(dataset_root, output_root, args.mm_emb_ids, args.batch_size)

    meta = {
        "source_dataset": str(dataset_root),
        "output_root": str(output_root),
        "copied_indexer": copied_indexer,
        "indexer_present": indexer is not None,
        "notes": [
            "seq/item/user ids remain RID as in TencentGR-1M training tables",
            "creative_emb keeps anonymous_cid/OID keys to match current sasrec load_mm_emb + indexer_i_rev lookup",
            "gr/train_data.json is a convenience export built directly from seq parquet",
        ],
        "counts": {
            "item_feat_items": len(item_feat_dict),
            "user_feat_users": len(user_feat_dict),
            **seq_stats,
        },
        "mm_emb": mm_stats,
    }
    write_json(output_root / "meta.json", meta)

    print("Bridge completed.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
