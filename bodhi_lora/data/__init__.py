"""Benchmark rows, the train, eval and probe pools, eval-id draws and the leakage gate.

    rows = load_rows(bench, data_root)               # loaders.py, labels.py, rubric.py
    split = split_rows(bench, cell.split, rows)      # splits.py
    ids = eval_ids(cell, split, rows_by_id, seed, inputs_dir)
    report = check_leakage(trained, split, held_texts, eval_ids_by_seed, population)
    report = check_adapter_leakage(trained, eval_texts)   # eval-only cells
"""
from bodhi_lora.data.leakage import LeakageError, check_adapter_leakage, check_leakage, normalize_text
from bodhi_lora.data.loaders import LOADERS, DataError, Row, load_rows, row_text, source_file
from bodhi_lora.data.splits import Split, SplitError, eval_ids, eval_population, read_id_list, split_rows

__all__ = ["LOADERS", "DataError", "LeakageError", "Row", "Split", "SplitError", "check_adapter_leakage",
           "check_leakage",
           "eval_ids", "eval_population", "load_rows", "normalize_text", "read_id_list", "row_text",
           "source_file", "split_rows"]
