# ECAPA experiment dataset v1

`data/processed/ecapa_experiment_v1` is a metadata-only experiment referencing
raw files under `data/audio`. No audio is copied or preprocessed at this stage.

## Global isolation

All paths in `data/metadata/asr_validation.csv` and
`data/metadata/asr_test.csv` are excluded before speaker selection. Command
audio lives outside `data/audio` and therefore cannot enter this dataset.

Audio ID, path, SHA-256 content, and task-group speaker isolation are validated.
Source metadata has no recording-session ID, so session-disjointness remains an
explicit limitation.

## Speaker Identification

Twenty closed-set speakers occur in all SID splits with distinct recordings:

- train: 400 recordings, 20 per speaker;
- validation: 100 recordings, 5 per speaker;
- test: 100 recordings, 5 per speaker.

## Speaker Verification

SV speakers never occur in SID. Validation and test speakers are also disjoint.

Each evaluation uses:

- three enrolled speakers with five enrollment and five positive-query files;
- two unknown speakers with ten negative-query files.

Each SV evaluation therefore has 50 recordings. Validation selects thresholds;
test remains sealed for final EER, FAR, FRR, and threshold evaluation.

## Build

The builder refuses overwrite. Build a candidate, verify it, then replace the
generated dataset intentionally:

```bash
python -m scripts.build_ecapa_experiment_dataset \
  --output-root data/processed/ecapa_experiment_v1.next
```

## Later ECAPA preprocessing

Raw files remain mono 48 kHz. Later preprocessing must create a new versioned
artifact: resample to 16 kHz, trim silence, normalize consistently, and apply
augmentation only to SID train. Validation/test audio must remain unchanged.

Embedding extraction must resolve metadata paths against the raw audio root:

```bash
python -m scripts.extract_all_embeddings \
  --metadata-dir data/processed/ecapa_experiment_v1/metadata \
  --audio-root data/audio \
  --protocol-file sid_train.csv \
  --protocol-file sid_validation.csv \
  --protocol-file sid_test.csv
```
