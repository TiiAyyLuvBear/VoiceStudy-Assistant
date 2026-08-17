"""Static tests for the Kaggle-input VoxVietnam preparation notebook."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "prepare-voxvietnam-on-kaggle.ipynb"


def _load() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(cell.get("source", "") for cell in notebook["cells"])
    return notebook, text


def test_notebook_is_self_contained_and_streams_kaggle_parquet() -> None:
    notebook, text = _load()

    assert notebook["nbformat"] == 4
    assert 'INPUT_ROOT = Path("/kaggle/input/voxvietnam-dataset")' in text
    assert 'INPUT_ROOT.glob("train_small-*.parquet")' in text
    assert 'INPUT_ROOT.glob("train-*.parquet")' in text
    assert 'INPUT_ROOT.glob("test-*.parquet")' in text
    assert 'load_dataset(\n            "parquet"' in text
    assert "streaming=True" in text
    assert "UserSecretsClient" not in text
    assert "manifest = materialize_voxvietnam_subset" in text
    assert "from src." not in text
    assert 'if __name__ == "__main__"' not in text


def test_notebook_enforces_compact_speaker_disjoint_protocol() -> None:
    _, text = _load()

    assert "TRAIN_SPEAKERS = 230" in text
    assert "VALIDATION_SPEAKERS = 50" in text
    assert "TEST_SPEAKERS = 50" in text
    assert "MAX_BYTES = 12 * 1024**3" in text
    assert "CLOSED_TRAIN_AUDIO = 20" in text
    assert "CLOSED_VALIDATION_AUDIO = 5" in text
    assert "OPEN_SET_KNOWN_SPEAKERS = 25" in text
    assert "protocols/closed_set/classifier_train.csv" in text
    assert "protocols/verification/validation_trials.csv" in text
    assert "protocols/open_set/test_queries.csv" in text
    assert 'source_manifest["invariants"]["speaker_disjoint"] is True' in text
    assert 'source_manifest["invariants"]["duplicate_checksum"] == 0' in text
    assert 'source_manifest["invariants"]["byte_budget_respected"] is True' in text
    assert "dataset_metadata.json" in text
    assert "file_manifest.csv" in text
    assert '"status": "complete" if package_complete else "incomplete"' in text
    assert 'print("Package incomplete. Missing required files:"' in text
    assert "shutil.make_archive" not in text


def test_notebook_orders_audit_qc_selection_and_materialization() -> None:
    _, text = _load()

    audit = text.index("def parquet_speaker_counts")
    sample_qc = text.index("sample_status = scan_audio_quality")
    full_qc = text.index("full_status = scan_audio_quality")
    selection = text.index("speaker_splits = select_speaker_splits")
    materialize = text.index("manifest = materialize_voxvietnam_subset")
    assert audit < sample_qc < full_qc < selection < materialize

    assert "def preprocess_audio" in text
    assert "for path in SOURCE_FILES[split]" in text
    assert "for row in source_summary.itertuples" not in text
    assert "duration_below_minimum" in text
    assert "duplicate_content" in text
    assert "resample_poly" in text
    assert 'subtype="PCM_16"' in text
    assert "source_selection_plan.csv" in text
    assert 'for split in ("train_small", "test")' in text
    assert 'source_records("train")' not in text


def test_all_python_cells_compile() -> None:
    notebook, _ = _load()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if source.lstrip().startswith("!"):
            continue
        compile(source, f"prepare-notebook-cell-{index}", "exec")


def test_preprocessing_cell_resamples_trims_and_rejects_bad_audio() -> None:
    notebook, _ = _load()
    source = next(
        cell["source"]
        for cell in notebook["cells"]
        if "def preprocess_audio" in cell.get("source", "")
    )
    namespace = {
        "Counter": Counter,
        "Mapping": Mapping,
        "SOURCE_FILES": {"train_small": [], "test": []},
        "INPUT_ROOT": Path("."),
        "TARGET_SAMPLE_RATE": 16_000,
        "MIN_DURATION_SEC": 2.0,
        "MAX_DURATION_SEC": 10.0,
        "SILENCE_TOP_DB": 35.0,
        "TRIM_PAD_SEC": 0.10,
        "source_summary": [],
        "np": np,
        "hashlib": hashlib,
    }
    exec(source, namespace)
    preprocess = namespace["preprocess_audio"]

    stereo = np.stack([
        np.full(8_000 * 3, 0.2, dtype=np.float32),
        np.full(8_000 * 3, 0.1, dtype=np.float32),
    ])
    result = preprocess({"array": stereo, "sampling_rate": 8_000})
    assert result["sample_rate"] == 16_000
    assert result["original_channels"] == 2
    assert result["waveform"].ndim == 1
    assert result["duration_sec"] == pytest.approx(3.0, abs=0.01)
    assert len(result["content_hash"]) == 64

    long_audio = np.full(16_000 * 12, 0.1, dtype=np.float32)
    assert preprocess({
        "array": long_audio,
        "sampling_rate": 16_000,
    })["duration_sec"] == pytest.approx(10.0)

    with pytest.raises(ValueError, match="silent_audio"):
        preprocess({
            "array": np.zeros(16_000 * 3, dtype=np.float32),
            "sampling_rate": 16_000,
        })
    with pytest.raises(ValueError, match="duration_below_minimum"):
        preprocess({
            "array": np.full(16_000, 0.1, dtype=np.float32),
            "sampling_rate": 16_000,
        })
