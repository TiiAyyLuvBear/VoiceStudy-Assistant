# ECAPA raw dataset

Source metadata for `data/audio` is `data/metadata/data_inventory.csv`:

- 2,148 metadata rows and matching WAV files;
- 2,036 rows marked valid;
- 34 speakers with valid audio;
- Vietnamese (`vi-VN`), mono, 48 kHz.

`data/datasets/ecapa_raw_v1` is a metadata-only closed-set SID dataset. Audio is
referenced from `data/audio`; nothing is copied or preprocessed.

## Build

```bash
python -m scripts.build_ecapa_raw_dataset
```

Selection rules:

- inventory row marked valid and referenced file exists;
- metadata duration from 2 through 10 seconds;
- mono 48 kHz according to inventory metadata;
- speaker has at least 30 eligible recordings;
- deterministic SHA-256 ordering with seed 42.

This yields 24 speakers and 720 recordings:

- train: 480, 20 per speaker;
- validation: 120, 5 per speaker;
- test: 120, 5 per speaker.

Every selected file receives an actual SHA-256 checksum. Splits have no audio
ID, path, or content overlap. Recording-session metadata is unavailable, so
session-disjointness cannot be guaranteed.

## Later preprocessing

Raw files remain 48 kHz. Before ECAPA embedding extraction, resample to 16 kHz,
trim silence, normalize consistently, and write a new versioned processed
dataset. Apply augmentation only to train. Keep validation and test unchanged.
