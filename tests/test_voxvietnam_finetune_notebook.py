"""Static contract tests for the Kaggle VoxVietnam fine-tuning notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT / "notebooks" / "finetune-ecapa-for-vietnamese-speaker-indentificat.ipynb"
)
EVALUATOR = ROOT / "src" / "speaker" / "evaluation.py"


def _notebook_text() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(
        cell.get("source", "")
        if isinstance(cell.get("source", ""), str)
        else "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    return notebook, text


def test_notebook_uses_voxvietnam_three_task_protocols() -> None:
    notebook, text = _notebook_text()
    evaluator_text = EVALUATOR.read_text(encoding="utf-8")

    assert notebook["nbformat"] == 4
    assert "voxvietnam_ecapa_three_task_v1" in text
    assert "prepare-voxvietnam-on-kaggle.ipynb" in text
    assert "protocols/verification/validation_trials.csv" in text
    assert "protocols/verification/test_trials.csv" in text
    assert "protocols/closed_set/classifier_train.csv" in text
    assert "protocols/closed_set/validation_queries.csv" in text
    assert "protocols/closed_set/test_queries.csv" in text
    assert "protocols/open_set/validation_gallery.csv" in text
    assert "protocols/open_set/test_queries.csv" in text
    assert 'speaker_sets["train"].isdisjoint' in text
    assert 'checkpoint_manager.save_if_improved("verification"' in text
    assert 'verification_threshold = validation_verification["min_dcf_threshold"]' in evaluator_text
    assert "open_threshold = select_open_set_threshold" in evaluator_text
    assert "from src.speaker.evaluation import" not in text
    assert "def evaluate_three_tasks(" in text
    assert "Attach repository source containing" not in text
    assert text.index('"frozen ECAPA-TDNN"') < text.index("for epoch in range(MAX_EPOCHS)")
    assert text.index('checkpoint = torch.load') < text.index(
        '"fine-tuned ECAPA-TDNN"'
    )


def test_notebook_caches_once_and_emits_three_task_artifact_contract() -> None:
    _, text = _notebook_text()
    evaluator_text = EVALUATOR.read_text(encoding="utf-8")

    assert 'evaluation_df["audio_path"].is_unique' in text
    assert "frozen_embedding_cache" in text
    assert "finetuned_embedding_cache" in text
    assert "evaluate_three_tasks(" in text
    assert text.count("evaluate_three_tasks(") == 3
    assert "def evaluate_three_tasks(" in evaluator_text
    for artifact in (
        "verification_metrics.json",
        "verification_trial_scores.csv",
        "closed_set_metrics.json",
        "closed_set_predictions.csv",
        "closed_set_confusion_matrix.csv",
        "open_set_metrics.json",
        "open_set_predictions.csv",
        "three_task_summary.csv",
    ):
        assert artifact in text or artifact in evaluator_text


def test_notebook_does_not_use_old_dataset_or_unseen_speaker_classifier() -> None:
    _, text = _notebook_text()

    assert "anyuuus/ecapa-sid-dataset" not in text
    assert "/kaggle/working/dataset" not in text
    assert "Test classifier" not in text
    assert "valid_macro_f1" not in text
    assert "voxvietnam_ecapa_verification_v1" not in text


def test_user_adjustable_configuration_lives_in_one_documented_cell() -> None:
    notebook, _ = _notebook_text()
    configurable = {
        "SEED",
        "EXPECTED_DATASET",
        "LABEL_COL",
        "TARGET_SAMPLE_RATE",
        "SEGMENT_SECONDS",
        "NUM_WORKERS",
        "GPU_COUNT",
        "PER_GPU_BATCH",
        "TARGET_EFFECTIVE_BATCH",
        "ENCODER_LEARNING_RATE",
        "CLASSIFIER_LEARNING_RATE",
        "WEIGHT_DECAY",
        "MAX_EPOCHS",
        "WARMUP_EPOCHS",
        "EARLY_STOPPING_PATIENCE",
        "WORKING_DIR",
        "CHECKPOINT_EVAL_INTERVAL",
        "MAX_ROLLING_CHECKPOINTS",
        "CHECKPOINT_DIR",
        "BEST_CLOSED_PATH",
        "BEST_VERIFICATION_PATH",
        "BEST_OPEN_PATH",
        "BEST_BALANCED_PATH",
        "LATEST_PATH",
        "MAX_CLOSED_F1_DROP",
    }
    config_markdown_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "markdown" and "Configuration — edit only this cell" in cell["source"]
    )
    config_cell_index = config_markdown_index + 1
    assert notebook["cells"][config_cell_index]["cell_type"] == "code"

    assignments: dict[str, list[int]] = {name: [] for name in configurable}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code" or cell["source"].lstrip().startswith("!"):
            continue
        tree = ast.parse(cell["source"])
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in assignments:
                    assignments[target.id].append(index)

    assert all(indices == [config_cell_index] for indices in assignments.values())


def test_all_python_cells_compile() -> None:
    notebook, _ = _notebook_text()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if source.lstrip().startswith("!"):
            continue
        compile(source, f"notebook-cell-{index}", "exec")


def test_notebook_defines_cross_cell_dependencies_before_use() -> None:
    _, text = _notebook_text()
    assert text.index("def protocol_vectors(") < text.index("frozen_closed_validation =")
    assert "summarize_three_task_results([frozen_results])" not in text
    assert text.index("for epoch in range(MAX_EPOCHS)") < text.index(
        'frozen_results = evaluate_three_tasks('
    )
