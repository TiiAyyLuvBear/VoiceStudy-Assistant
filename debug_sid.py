from src.speaker.identification import identify_closed_set_svm
from src.speaker.application_identification import identify_application_user
from src.speaker.verification import verify_speaker

audio = "data/commands/audio/validation/REC_VAL0004_cmdspk01.wav"

print("=== Closed-set SVM ===")
print(identify_closed_set_svm(audio))

print("\n=== Application SID ===")
print(identify_application_user(audio))

print("\n=== Speaker Verification ===")
print(verify_speaker(audio, "user_001"))