# ECAPA-TDNN Product-Use Considerations

## Purpose and current scope

The fine-tuned ECAPA-TDNN checkpoint is acceptable for a student-coursework
demonstration in a controlled, low-noise environment. It should currently be
treated as an experimental model, not as a production-grade authentication
mechanism or the sole control protecting sensitive operations.

The intended short-term use is speaker verification with five enrollment
recordings per user, consistent recording equipment, a quiet environment, and
a retry or fallback path when a genuine user is rejected.

## Current evidence

The authoritative detailed evaluation artifacts under
`reports/results/evaluation/finetuned/` report:

- Test EER: `15.0%`.
- Test minDCF: `0.5196`.
- Validation-selected verification threshold: `0.43221902`.
- Test FAR at that threshold: `0.28%`.
- Test FRR at that threshold: `36.4%`.
- Test TAR at that threshold: `63.6%`.
- Evaluation size: 3,000 test trials, consisting of 500 genuine and 2,500
  impostor trials.

Fine-tuning improves verification relative to the frozen ECAPA baseline, but
the large change from `5.2%` validation EER to `15.0%` test EER indicates a
generalization gap. The model may perform worse when microphone, room, noise,
speaker condition, or recording session differs from the training data.

## Acceptable coursework use

- Demonstration or evaluation with cooperative users.
- Quiet or lightly noisy rooms.
- Similar microphones and speaking distances during enrollment and testing.
- Low-risk personalization where a false decision causes no material harm.
- Flows that permit two or three retries and provide a non-voice fallback.

## Uses to avoid for now

- Password replacement or voice-only login.
- Payments, confidential-data access, or administrative authorization.
- Attendance, identity, or disciplinary decisions without human review.
- Adversarial environments involving replayed or synthesized speech.
- Claims that the model is production-ready or robust across devices and
  environments.

## Requirements when integrating the checkpoint

1. Use the same fine-tuned encoder for both enrollment and verification.
2. Recreate all enrollment centroids after changing the encoder checkpoint.
3. Keep preprocessing consistent: mono audio, 16 kHz sample rate, identical
   normalization, L2-normalized embeddings, and a normalized mean enrollment
   centroid.
4. Use a threshold selected with validation data produced by the same encoder.
   Never select or adjust the threshold using test results.
5. Bind the checkpoint hash, preprocessing version, embedding dimension, and
   threshold version together. Reject enrollment vectors created by another
   model version.
6. Reject silence, clipping, corrupted audio, and recordings below the chosen
   duration or quality limits.
7. Provide retry and PIN/password/manual fallback paths because the current
   false-rejection rate is high.
8. Log similarity, decision, model version, threshold version, audio-quality
   status, and latency without retaining raw voice recordings unnecessarily.

## Required improvements before production use

- Collect in-domain data across multiple days, microphones, rooms, distances,
  noise levels, and speaker conditions.
- Evaluate on fully unseen users and sessions that match the deployed system.
- Add noise, reverberation, codec, gain, speed, and microphone augmentation.
- Recalibrate the operating threshold against an explicit acceptable FAR and
  FRR for the product rather than relying on EER alone.
- Measure confidence intervals and performance by device, environment, and
  speaker subgroup.
- Add replay, synthesized-voice, and deepfake resistance through anti-spoofing
  or a challenge-response phrase.
- Run a real-user pilot and document failure cases before broader release.

For a normal low-risk assistant feature, a reasonable future target is an
in-domain held-out EER below `5%`, FRR below `10%` at the selected security
operating point, and a small validation-to-test performance gap. These are
project targets, not universal production guarantees.

## Reporting caveat

`reports/three_task_summary.csv` and
`reports/results/three_task_summary.csv` contain different metric values. Use
the detailed metrics and trial-score files from one reproducible run as the
source of truth, then regenerate all summaries from that run before presenting
final results.

## Current release decision

**Approved for controlled student-coursework demonstration with retries and a
fallback. Not approved for production authentication or security-sensitive
decisions.**

## Integration status

The application now loads the fine-tuned epoch-9 encoder from
`reports/results/ecapa_voxvietnam_best.pt`. `config.yaml` binds the checkpoint
SHA-256, encoder key, model version, preprocessing contract, five-recording
enrollment rule, and validation-selected verification threshold. Enrollment,
SID, and SV share one cached encoder.

Existing application centroids created by the baseline encoder are deliberately
rejected with `CENTROID_MODEL_MISMATCH`. Re-enroll every user before running a
speaker-dependent demonstration.

Automated quality regression checks reproduce the 3,000 held-out trial results:
EER `15.0%`, minDCF `0.5196`, FAR `0.28%`, and FRR `36.4%`. These checks prevent
accidental artifact or threshold drift; they do not improve the model's known
real-world limitations.
