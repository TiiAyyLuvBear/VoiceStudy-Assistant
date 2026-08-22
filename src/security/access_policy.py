"""Intent access rules shared by UI and pipeline."""

PUBLIC = "PUBLIC"
SID = "SID"
SID_AND_SV = "SID_AND_SV"
REJECT = "REJECT"

INTENT_POLICIES = {
    "GET_TIME": PUBLIC,
    "VIEW_SCHEDULE": SID,
    "ADD_SCHEDULE": SID,
    "ADD_NOTE": SID,
    "ADD_PRIVATE_NOTE": SID_AND_SV,
    "VIEW_PRIVATE_NOTE": SID_AND_SV,
    "OUT_OF_SCOPE": REJECT,
}


def get_access_policy(intent: str) -> str:
    return INTENT_POLICIES.get(intent, REJECT)
