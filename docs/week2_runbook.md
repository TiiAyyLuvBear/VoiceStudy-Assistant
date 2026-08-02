# Week 2 ASR/NLU validation runbook

## Outputs that do not require command audio

Refresh the ASR prediction, summary, and required metrics filename:

```powershell
python -m scripts.evaluate_asr data/metadata/asr_validation.csv
```

This produces:

- `reports/asr/asr_validation_predictions.csv`
- `reports/asr/asr_validation_summary.json`
- `reports/asr/asr_validation_metrics.json`

## Partial command-audio run

While audio is incomplete, generate a transparent partial report:

```powershell
python -m scripts.evaluate_week2_nlu --allow-incomplete
```

Pending commands remain in the output with `evaluated=false` and
`error=not_recorded`. No fake prediction is generated.

## Official run after all validation audio is recorded

```powershell
python -m scripts.validate_command_audio --split validation
python -m scripts.evaluate_week2_nlu
```

Strict mode exits with code 2 if any of the 30 validation recordings is
missing. Successful existing rows are resumed, so only new audio is sent to
Whisper.

The command creates:

- `reports/nlu/intent_validation_ground_truth.csv`
- `reports/nlu/intent_validation_whisper.csv`
- `reports/nlu/out_of_scope_validation.csv`
- `reports/nlu/entity_validation_results.csv`
- `reports/nlu/week2_validation_summary.json`

The official result is complete only when `complete=true`,
`audio_ready_count=30`, and `audio_evaluated_count=30`.
