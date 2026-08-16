# ASR fine-tune dataset v3

This dataset allocates every usable row from
`data/metadata/data_inventory.csv` without modifying the frozen v1/v2
artifacts.

## Files

- `metadata/asr_finetune_train.csv`: model training only.
- `metadata/asr_finetune_validation.csv`: hyperparameter/model selection.
- `metadata/asr_finetune_test.csv`: final evaluation only, after configuration
  and model lock.
- `metadata/asr_finetune_rejected.csv`: unusable source rows and their reasons.
- `asr_finetune_manifest.json`: provenance, split rules, counts, and checksums.

The training columns needed by an ASR trainer are `audio_path` and
`transcript`. `audio_sha256` pins each referenced audio file to its current
content. Transcripts are Unicode NFC-normalized and have repeated whitespace
collapsed.

For Hugging Face Datasets, load and cast the audio column as follows:

```python
from datasets import Audio, load_dataset

dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/processed/v3/metadata/asr_finetune_train.csv",
        "validation": "data/processed/v3/metadata/asr_finetune_validation.csv",
        "test": "data/processed/v3/metadata/asr_finetune_test.csv",
    },
)
dataset = dataset.rename_column("audio_path", "audio")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Use the existing `transcript` column as the text label. Do not train on or tune
against the test CSV.

## Frozen result

ASR v3 is now `FROZEN`. Whisper Small was fine-tuned with decoder-only LoRA;
epoch 2 was selected using validation loss before the locked test was run once.
On the 249-sample test, WER improved from 23.211% to 20.000% and CER improved
from 14.624% to 12.007%. See `reports/asr/v3/comparison.json` and
`models/experimental/asr/v3/final_manifest.json` for checksums and provenance.

Do not rebuild or modify this v3 dataset. Create v4 for future data/model work.

Rebuild from the project root with:

```powershell
python -m scripts.build_asr_finetune_dataset
```
