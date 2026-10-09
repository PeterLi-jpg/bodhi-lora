"""Benchmark rows from a benchmark record: fetch or read the source, build each row,
and cache the result under the data root.

A row is one prompt with its rubric. On disk it is one JSON line,

    {"prompt_id", "prompt": [chat turns], "rubrics": [{"criterion", "points", "tags"}],
     "example_tags": [...]}

the format of the rubric JSONL sources, so every stage reads every benchmark the same
way. In memory it is a Row, which also carries the row's labels and where it sits in
each source file.

Loaders are named by source format, not by dataset (LOADERS):
  rubric_jsonl  rows as given in one or more JSONL files (a url or local source)
  hf_qa         question-answer rows of a Hugging Face dataset at a pinned commit; the
                record's `fields` map the loader's fields to source columns, and every
                row gets the synthesized rubric (bodhi_lora/data/rubric.py)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bodhi_lora.data.labels import label_value
from bodhi_lora.data.rubric import SHARED_CRITERIA, synthesized_rubric
from bodhi_lora.paths import REPO_ROOT
from bodhi_lora.registry.schema import BenchmarkConfig, HfSource, LocalSource, UrlSource

# The position key of rows from a source that has no named files (a Hugging Face split).
SOURCE = "source"


@dataclass(frozen=True)
class Row:
    prompt_id: str
    messages: list = field(compare=False)       # chat turns up to the turn to answer
    rubrics: list = field(compare=False)        # {"criterion", "points", "tags"}
    tags: list = field(compare=False)
    labels: dict = field(compare=False)         # label name -> True, False or None
    pair_id: Optional[str] = None               # rows of one pair share it (paired probes)
    # Source file key -> the row's index in that file, so a split can keep file order.
    positions: dict = field(default_factory=dict, compare=False)

    def to_json(self) -> dict:
        return {"prompt_id": self.prompt_id, "prompt": self.messages, "rubrics": self.rubrics,
                "example_tags": self.tags}


class DataError(RuntimeError):
    """A benchmark source is missing, changed, or could not be built."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _with_labels(bench: BenchmarkConfig, row: dict, positions: dict) -> Row:
    labels = {name: label_value(rule, row.get("example_tags", []), row.get("rubrics", []), row)
              for name, rule in bench.labels.items()}
    return Row(prompt_id=row["prompt_id"], messages=row["prompt"], rubrics=row["rubrics"],
               tags=list(row.get("example_tags", [])), labels=labels, positions=positions)


# -- rubric JSONL sources --------------------------------------------------------------

def source_file(bench: BenchmarkConfig, key: str, data_root: Path, fetch: bool = True) -> Path:
    """The local copy of one source file, checked against its sha256.

    A url file is downloaded to <data_root>/<benchmark>/<key>.jsonl when it is missing
    and fetch is true. A local file is read from the repository.
    """
    src = bench.source
    if isinstance(src, LocalSource):
        f = src.files[key]
        path = REPO_ROOT / f.path
    elif isinstance(src, UrlSource):
        f = src.files[key]
        path = Path(data_root) / bench.id / f"{key}.jsonl"
        if not path.is_file():
            if not fetch:
                raise DataError(f"{path} is missing; run python -m bodhi_lora fetch {bench.id}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            print(f"downloading {bench.id} {key} -> {path}", flush=True)
            with urllib.request.urlopen(f.url) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out)
            os.replace(tmp, path)
    else:
        raise DataError(f"{bench.id}: source kind {src.kind} has no files")
    got = sha256_file(path)
    if got != f.sha256:
        raise DataError(f"{path}: sha256 {got} is not the pinned {f.sha256}; the source changed")
    return path


def load_rubric_jsonl(bench: BenchmarkConfig, data_root: Path, fetch: bool = True) -> list[Row]:
    """Rows of every source file, each prompt id once, in order of first appearance."""
    rows: dict[str, dict] = {}
    positions: dict[str, dict] = {}
    for key in bench.source.files:
        path = source_file(bench, key, data_root, fetch)
        with open(path, encoding="utf-8") as fh:
            index = 0
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DataError(f"{path}:{lineno}: malformed JSON") from e
                pid = row["prompt_id"]
                rows.setdefault(pid, row)
                positions.setdefault(pid, {})[key] = index
                index += 1
    return [_with_labels(bench, row, positions[pid]) for pid, row in rows.items()]


# -- Hugging Face question-answer sources ---------------------------------------------

class HfQaParams(BaseModel):
    """loader_params of an hf_qa benchmark."""

    model_config = ConfigDict(extra="forbid")
    # The text the prompt id hashes: the question as stored, or with outer spaces
    # removed. It must stay as the first runs had it, or their eval ids stop matching.
    id_text: Literal["raw", "stripped"]
    reference_max_chars: int = Field(gt=0)
    # space: newlines in the reference become spaces before it is cut.
    reference_newlines: Literal["keep", "space"]
    # Single-best-answer sources that store the answer as an index into choice columns.
    choice_fields: list[str] = Field(default_factory=list)
    choice_index_field: Optional[str] = None


def _first(ex: dict, keys: list[str], default: Any = "") -> Any:
    """The first present, non-empty value among the candidate columns."""
    for k in keys:
        if k in ex and ex[k] not in (None, "", [], {}):
            return ex[k]
    return default


def stable_id(prefix: str, text: str) -> str:
    """Prompt id from the question text, so a rebuild gives the same ids."""
    return f"{prefix}-{hashlib.sha1(f'{prefix}:{text}'.encode('utf-8')).hexdigest()[:12]}"


def build_hf_qa_rows(bench: BenchmarkConfig, source_rows) -> tuple[list[dict], dict]:
    """Rows in the rubric JSONL format from question-answer source rows, and counts.

    A row without a question or an answer is unusable; a row whose prompt id is already
    taken is a duplicate. Both are counted and left out.
    """
    p = HfQaParams.model_validate(bench.loader_params)
    fields = bench.fields
    rows, seen = [], set()
    counts = {"n_source_rows": 0, "n_unusable": 0, "n_duplicates": 0}
    for ex in source_rows:
        counts["n_source_rows"] += 1
        question = _first(ex, fields["question"])
        if p.choice_fields:
            choices = [_first(ex, [k]) for k in p.choice_fields]
            index = ex.get(p.choice_index_field)
            answer = None
            if question and index is not None and any(choices):
                try:
                    answer = choices[int(index)]
                except (ValueError, TypeError, IndexError):
                    answer = None
        else:
            answer = _first(ex, fields["answer"])
            # A source may store the answer as an option key; the option text replaces it.
            options = _first(ex, fields.get("options", []), default=None)
            if isinstance(options, dict) and isinstance(answer, str) and answer in options:
                answer = options[answer]
        if not question or not answer:
            counts["n_unusable"] += 1
            continue
        text = str(question).strip()
        pid = stable_id(bench.id_prefix, question if p.id_text == "raw" else text)
        if pid in seen:
            counts["n_duplicates"] += 1
            continue
        seen.add(pid)
        ref = str(answer).strip()
        if p.reference_newlines == "space":
            ref = ref.replace("\n", " ")
        ref = ref[:p.reference_max_chars]
        rows.append({
            "prompt_id": pid,
            "prompt": [{"role": "user", "content": bench.prompt_template.format(question=text)}],
            "rubrics": synthesized_rubric(bench.reference_criterion.format(reference=ref)),
            "example_tags": list(bench.tags),
        })
    counts["n_written"] = len(rows)
    return rows, counts


def hf_qa_build_key(bench: BenchmarkConfig) -> str:
    """sha256 of every setting a built row depends on, so a cache never outlives them."""
    keyed = {"source": bench.source.model_dump(), "fields": bench.fields,
             "loader_params": bench.loader_params, "id_prefix": bench.id_prefix,
             "prompt_template": bench.prompt_template,
             "reference_criterion": bench.reference_criterion, "tags": bench.tags,
             "shared_criteria": SHARED_CRITERIA}
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()


def _hf_rows(src: HfSource):
    # Imported here: only a build needs datasets, and the training env may lack it.
    from datasets import load_dataset
    return load_dataset(src.dataset, src.config_name, split=src.split, revision=src.revision)


def load_hf_qa(bench: BenchmarkConfig, data_root: Path, fetch: bool = True,
               source_rows: Optional[Callable] = None) -> list[Row]:
    """Rows built from the pinned dataset, cached in <data_root>/<benchmark>/rows.jsonl.

    The cache carries the build key; a cache built with other settings is rebuilt.
    source_rows replaces the download (tests).
    """
    folder = Path(data_root) / bench.id
    cache, meta_path = folder / "rows.jsonl", folder / "rows.manifest.json"
    key = hf_qa_build_key(bench)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if not (cache.is_file() and meta.get("build_key") == key
            and meta.get("sha256") == sha256_file(cache)):
        if not fetch:
            raise DataError(f"{cache} is missing or stale; run python -m bodhi_lora fetch {bench.id}")
        rows, counts = build_hf_qa_rows(bench, (source_rows or _hf_rows)(bench.source))
        if not rows:
            raise DataError(f"{bench.id}: no usable rows; check the record's fields against the source columns")
        folder.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        os.replace(tmp, cache)
        meta = {"benchmark": bench.id, "build_key": key, "source": bench.source.model_dump(),
                "counts": counts, "sha256": sha256_file(cache)}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    out = []
    with open(cache, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            out.append(_with_labels(bench, json.loads(line), {SOURCE: index}))
    return out


LOADERS: dict[str, Callable[..., list[Row]]] = {
    "rubric_jsonl": load_rubric_jsonl,
    "hf_qa": load_hf_qa,
}


def load_rows(bench: BenchmarkConfig, data_root: Path, fetch: bool = True) -> list[Row]:
    """Every row of a benchmark, by its record's loader. fetch=False never downloads."""
    if bench.loader not in LOADERS:
        raise DataError(f"{bench.id}: no loader {bench.loader!r}; loaders are {sorted(LOADERS)}")
    return LOADERS[bench.loader](bench, Path(data_root), fetch=fetch)


def row_text(row: Row) -> str:
    """The last user turn, lowercased with spaces collapsed: what the leakage gate compares."""
    for m in reversed(row.messages or []):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
            return " ".join(str(content).lower().split())
    return ""
