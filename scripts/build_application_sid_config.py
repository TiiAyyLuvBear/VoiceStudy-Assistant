"""Freeze the application speaker-identification configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.files import sha256_file


APPLICATION_THRESHOLD = (
    PROJECT_ROOT
    / "models"
    / "application"
    / "application_sid_threshold.json"
)

APPLICATION_VERIFICATION_THRESHOLD = (
    PROJECT_ROOT
    / "models"
    / "application"
    / "application_verification_threshold.json"
)

APPLICATION_EMBEDDINGS_DIR = (
    PROJECT_ROOT
    / "models"
    / "application"
    / "user_embeddings"
)

OUTPUT_CONFIG = (
    PROJECT_ROOT
    / "models"
    / "application"
    / "application_sid_config.json"
)

ECAPA_DIR = (
    PROJECT_ROOT
    / "models"
    / "cache"
    / "ecapa"
)

PREPROCESSING_VERSION = "v1"

RANDOM_SEED = 42


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main() -> int:
    # ---------------------------------------------------------
    # 1. Load frozen experimental threshold
    # ---------------------------------------------------------

    threshold_config = load_json(APPLICATION_THRESHOLD)

    if "threshold" not in threshold_config:
        raise ValueError(
            "Experimental threshold config does not contain 'threshold'"
        )

    threshold = float(threshold_config["threshold"])

    verification_config = load_json(
        APPLICATION_VERIFICATION_THRESHOLD
    )
    if "threshold" not in verification_config:
        raise ValueError(
            "Verification threshold config does not contain 'threshold'"
        )
    verification_threshold = float(
        verification_config["threshold"]
    )

    # ---------------------------------------------------------
    # 2. Check application enrollment embeddings
    # ---------------------------------------------------------

    if not APPLICATION_EMBEDDINGS_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing application embedding directory: "
            f"{APPLICATION_EMBEDDINGS_DIR}"
        )

    embedding_files = sorted(
        APPLICATION_EMBEDDINGS_DIR.glob("*.npy")
    )

    if not embedding_files:
        raise ValueError(
            "No application user embeddings found."
        )

    # ---------------------------------------------------------
    # 3. Check ECAPA checkpoint
    # ---------------------------------------------------------

    ecapa_embedding_model = ECAPA_DIR / "embedding_model.ckpt"
    ecapa_hyperparameters = ECAPA_DIR / "hyperparams.yaml"

    if not ecapa_embedding_model.is_file():
        raise FileNotFoundError(
            f"Missing ECAPA embedding checkpoint: "
            f"{ecapa_embedding_model}"
        )
    if not ecapa_hyperparameters.is_file():
        raise FileNotFoundError(
            f"Missing ECAPA hyperparameters: {ecapa_hyperparameters}"
        )

    # ---------------------------------------------------------
    # 4. Build frozen application configuration
    # ---------------------------------------------------------

    config = {
        "protocol": "APPLICATION_SID",

        "model": {
            "embedding_model": "ECAPA-TDNN",
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "checkpoint": ecapa_embedding_model.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "checkpoint_sha256": sha256_file(ecapa_embedding_model),
            "hyperparameters": ecapa_hyperparameters.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "hyperparameters_sha256": sha256_file(
                ecapa_hyperparameters
            ),
            "embedding_dim": 192,
        },

        "preprocessing": {
            "version": PREPROCESSING_VERSION,
        },

        "decision": {
            "method": "cosine_centroid",
            "threshold": threshold,
            "comparison": (
                "cosine_similarity >= threshold means KNOWN"
            ),
        },

        "threshold_source": {
            "type": threshold_config.get("selection"),
            "source": APPLICATION_THRESHOLD.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "selection_criterion": threshold_config.get(
                "selection_criterion"
            ),
        },

        "speaker_verification": {
            "method": "cosine_centroid",
            "threshold": verification_threshold,
            "comparison": (
                "cosine_similarity >= threshold means VERIFIED"
            ),
            "threshold_source": {
                "type": verification_config.get("selection"),
                "source": (
                    APPLICATION_VERIFICATION_THRESHOLD.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
                "selection_criterion": verification_config.get(
                    "selection_criterion"
                ),
            },
        },

        "application_enrollment": {
            "directory": APPLICATION_EMBEDDINGS_DIR.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "embedding_count": len(embedding_files),
            "embedding_files": [
                path.relative_to(PROJECT_ROOT).as_posix()
                for path in embedding_files
            ],
            "required_audio_per_user": 5,
            "speaker_model_training": False,
        },

        "random_seed": RANDOM_SEED,

        "status": "FROZEN",
    }

    # ---------------------------------------------------------
    # 5. Write configuration
    # ---------------------------------------------------------

    OUTPUT_CONFIG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_CONFIG.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("APPLICATION SID CONFIGURATION FROZEN")
    print("=" * 60)
    print(
        f"Output    : "
        f"{OUTPUT_CONFIG.relative_to(PROJECT_ROOT).as_posix()}"
    )
    print(f"Threshold : {threshold:.8f}")
    print(f"SV threshold: {verification_threshold:.8f}")
    print(
        f"Embeddings: {len(embedding_files)}"
    )
    print(
        f"Checkpoint: "
        f"{ecapa_embedding_model.relative_to(PROJECT_ROOT).as_posix()}"
    )
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
