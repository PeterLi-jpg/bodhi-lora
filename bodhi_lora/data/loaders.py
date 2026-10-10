"""Benchmark rows from a benchmark record: fetch or read the source, build each row,
and cache the result under the data root.

A row is one prompt with its rubric. On disk it is one JSON line,

    {"prompt_id", "prompt": [chat turns], "rubrics": [{"criterion", "points", "tags"}],
     "example_tags": [...]}

the format of the rubric JSONL sources, so every stage reads every benchmark the same
way. In memory it is a Row, which also carries the row's labels and where it sits in
each source file.

Loaders are named by source format, not by dataset (LOADERS):
  rubric_jsonl          rows as given in one or more JSONL files (a url or local source)
  hf_qa                 question-answer rows of a Hugging Face dataset at a pinned commit;
                        the record's `fields` map the loader's fields to source columns,
                        and every row gets the synthesized rubric (bodhi_lora/data/rubric.py)
  paired_context_jsonl  JSONL cases whose context is a list of sentences; each case gives
                        a pair of rows, one with every sentence and one with the first few
                        only, so whether information is missing is known by construction
  xml_records           XML records, one per row, fields read by element path; a file may
                        hold one document or many documents one after another
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
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
    """A Row from a row dict. Field labels read the dict's own keys (e.g. "variant"), and
    a paired loader's "pair_id" key becomes Row.pair_id."""
    labels = {name: label_value(rule, row.get("example_tags", []), row)
              for name, rule in bench.labels.items()}
    return Row(prompt_id=row["prompt_id"], messages=row["prompt"], rubrics=row["rubrics"],
               tags=list(row.get("example_tags", [])), labels=labels, pair_id=row.get("pair_id"),
               positions=positions)


# -- rubric JSONL sources --------------------------------------------------------------

def source_file(bench: BenchmarkConfig, key: str, data_root: Path, fetch: bool = True) -> Path:
    """The local copy of one source file, checked against its sha256.

    A url file is downloaded to <data_root>/<benchmark>/<key><suffix of the url> when it
    is missing and fetch is true. A local file is read from the repository.
    """
    src = bench.source
    if isinstance(src, LocalSource):
        f = src.files[key]
        path = REPO_ROOT / f.path
    elif isinstance(src, UrlSource):
        f = src.files[key]
        # The URL's own suffix (.jsonl, .xml), so the local copy says what it holds.
        path = Path(data_root) / bench.id / f"{key}{Path(urlparse(f.url).path).suffix or '.jsonl'}"
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
    if getattr(f, "sha256", None) is None:
        # An unpinned file (UrlFile.unpinned_reason): say what to pin; the run's inputs
        # manifest records the hash it read (source_hashes).
        print(f"{bench.id} {key}: unpinned; its sha256 is {got}. Pin it in "
              f"configs/benchmarks/{bench.id}.yaml.", flush=True)
    elif got != f.sha256:
        raise DataError(f"{path}: sha256 {got} is not the pinned {f.sha256}; the source changed")
    return path


def source_hashes(bench: BenchmarkConfig, data_root: Path) -> dict:
    """What a run read: the sha256 of each local copy of a file source, or the pinned
    revision of a Hugging Face source."""
    if isinstance(bench.source, HfSource):
        return {"revision": bench.source.revision}
    return {key: sha256_file(source_file(bench, key, data_root, fetch=False)) for key in bench.source.files}


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


# -- paired cases: every context sentence against the first few ----------------------------

class PairedContextParams(BaseModel):
    """loader_params of a paired_context_jsonl benchmark."""

    model_config = ConfigDict(extra="forbid")
    # Sentences the incomplete row keeps from the start of the context. MediQ's initial
    # information is the first sentence (src/patient.py: initial_info = context[0]).
    initial_sentences: int = Field(default=1, gt=0)
    reference_max_chars: int = Field(gt=0)


VARIANTS = ("complete", "incomplete")


def build_paired_rows(bench: BenchmarkConfig, cases) -> tuple[list[dict], dict]:
    """Two rows per case, in the rubric JSONL format plus "pair_id" and "variant".

    The complete row's prompt has every context sentence, the incomplete row's the first
    initial_sentences; both end with the case's question. The pair id hashes the complete
    prompt, so it does not depend on the source order. A case without a question, an
    answer, or more context than the incomplete row keeps is counted and left out: its
    two rows would not differ in what they leave out.
    """
    p = PairedContextParams.model_validate(bench.loader_params)
    fields = bench.fields
    rows, seen = [], set()
    counts = {"n_source_rows": 0, "n_unusable": 0, "n_too_little_context": 0, "n_duplicates": 0}
    for case in cases:
        counts["n_source_rows"] += 1
        question, answer = _first(case, fields["question"]), _first(case, fields["answer"])
        context = _first(case, fields["context"], default=[])
        if isinstance(context, str):
            context = [context]
        context = [str(c).strip() for c in context if str(c).strip()]
        if not question or not answer:
            counts["n_unusable"] += 1
            continue
        if len(context) <= p.initial_sentences:
            counts["n_too_little_context"] += 1
            continue
        texts = {"complete": " ".join(context), "incomplete": " ".join(context[:p.initial_sentences])}
        prompts = {v: bench.prompt_template.format(context=t, question=str(question).strip())
                   for v, t in texts.items()}
        pair_id = stable_id(bench.id_prefix, prompts["complete"])
        if pair_id in seen:
            counts["n_duplicates"] += 1
            continue
        seen.add(pair_id)
        rubric = synthesized_rubric(bench.reference_criterion.format(
            reference=str(answer).strip()[:p.reference_max_chars]))
        for variant in VARIANTS:
            rows.append({"prompt_id": f"{pair_id}-{variant}", "pair_id": pair_id, "variant": variant,
                         "prompt": [{"role": "user", "content": prompts[variant]}],
                         "rubrics": [dict(c, tags=list(c["tags"])) for c in rubric],
                         "example_tags": list(bench.tags) + [f"variant:{variant}"]})
    counts["n_pairs"] = len(seen)
    return rows, counts


def _jsonl_records(bench: BenchmarkConfig, data_root: Path, fetch: bool):
    for key in bench.source.files:
        path = source_file(bench, key, data_root, fetch)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DataError(f"{path}:{lineno}: malformed JSON") from e


def load_paired_context_jsonl(bench: BenchmarkConfig, data_root: Path, fetch: bool = True) -> list[Row]:
    rows, counts = build_paired_rows(bench, _jsonl_records(bench, data_root, fetch))
    if not rows:
        raise DataError(f"{bench.id}: no usable pairs ({counts}); check the record's fields")
    return [_with_labels(bench, row, {SOURCE: i}) for i, row in enumerate(rows)]


# -- XML records -------------------------------------------------------------------------

class XmlParams(BaseModel):
    """loader_params of an xml_records benchmark. Paths are ElementTree paths below a record."""

    model_config = ConfigDict(extra="forbid")
    record_tag: str
    id_path: str
    # Tag name -> element path; each non-empty value becomes the row tag "<name>:<value>".
    tag_paths: dict[str, str] = Field(default_factory=dict)


_XML_DECLARATION = re.compile(r"<\?xml[^>]*\?>")


def parse_xml_records(text: str, record_tag: str) -> list:
    """The record elements of a file that holds one XML document or many in a row.

    Some sources write each record as its own document, each with an XML declaration.
    The declarations go, and the rest is parsed inside one root element. The sources are
    pinned by sha256; the expat parser that ElementTree uses (2.4 and later) also limits
    entity expansion.
    """
    body = _XML_DECLARATION.sub("", text)
    root = ET.fromstring(f"<records>{body}</records>")
    return root.iter(record_tag)


def _text(record, path: str) -> str:
    node = record.find(path)
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def build_xml_rows(bench: BenchmarkConfig, records) -> tuple[list[dict], dict]:
    """Rows in the rubric JSONL format, with no rubric (rubric: none)."""
    p = XmlParams.model_validate(bench.loader_params)
    rows, seen = [], set()
    counts = {"n_source_rows": 0, "n_unusable": 0, "n_duplicates": 0}
    for rec in records:
        counts["n_source_rows"] += 1
        question = next((t for t in (_text(rec, path) for path in bench.fields["question"]) if t), "")
        rid = _text(rec, p.id_path)
        if not question or not rid:
            counts["n_unusable"] += 1
            continue
        if rid in seen:
            counts["n_duplicates"] += 1
            continue
        seen.add(rid)
        tags = list(bench.tags) + [f"{name}:{v}" for name, path in p.tag_paths.items()
                                   for v in [_text(rec, path)] if v]
        rows.append({"prompt_id": rid, "rubrics": [], "example_tags": tags,
                     "prompt": [{"role": "user", "content": bench.prompt_template.format(question=question)}]})
    counts["n_written"] = len(rows)
    return rows, counts


def load_xml_records(bench: BenchmarkConfig, data_root: Path, fetch: bool = True) -> list[Row]:
    p = XmlParams.model_validate(bench.loader_params)
    records = []
    for key in bench.source.files:
        path = source_file(bench, key, data_root, fetch)
        records += list(parse_xml_records(path.read_text(encoding="utf-8"), p.record_tag))
    rows, counts = build_xml_rows(bench, records)
    if not rows:
        raise DataError(f"{bench.id}: no usable records ({counts}); check record_tag, id_path and fields")
    return [_with_labels(bench, row, {SOURCE: i}) for i, row in enumerate(rows)]


LOADERS: dict[str, Callable[..., list[Row]]] = {
    "rubric_jsonl": load_rubric_jsonl,
    "hf_qa": load_hf_qa,
    "paired_context_jsonl": load_paired_context_jsonl,
    "xml_records": load_xml_records,
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
