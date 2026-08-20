import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from src.nlu.intent_classifier import classify_intent, _classify_regex, FUZZY_CANDIDATES, KEYWORD_ANCHORS
from src.nlu.text_normalizer import normalize_text
from src.utils.fuzzy_match import fuzzy_match, normalize_for_matching, _score_pair

cases = [
    ("REC_TST0005", "GET_TIME",            "bây giờ thời gian thế nào"),
    ("REC_TST0009", "VIEW_SCHEDULE",       "Ngày mai mình có bối gì"),
    ("REC_TST0013", "ADD_SCHEDULE",        "Làm cuộc hẹn gặp có vấn lúc chính giờ sáng ngày kia"),
    ("REC_TST0016", "VIEW_PRIVATE_NOTE",   "Xin đọc ghi chủ riêng tư của tôi"),
    ("REC_TST0017", "VIEW_PRIVATE_NOTE",   "mở nót bảo mật gần nhất"),
    ("REC_TST0018", "VIEW_PRIVATE_NOTE",   "cho tôi xem ghi chủ cả nhân mới nhất"),
    ("REC_TST0019", "VIEW_PRIVATE_NOTE",   "hiển hệ not ring tư của tôi"),
    ("REC_TST0030", "OUT_OF_SCOPE",        "hôm nay có sự kiện gì"),
    # Also check the problem case
    ("existing",    "OUT_OF_SCOPE",        "Mở ghi chú"),
]

print(f"{'ID':<14} {'Expected':<20} {'Result':<20} {'OK?':>5}")
print("-" * 65)

for rid, expected, transcript in cases:
    result = classify_intent(transcript)
    ok = "OK" if result == expected else "MISS"
    print(f"{rid:<14} {expected:<20} {result:<20} {ok:>5}")
