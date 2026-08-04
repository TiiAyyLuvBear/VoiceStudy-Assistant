"""Create safe demo records for Week 1 manual testing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.database import create_database
from src.database.user_repository import create_user, get_user
from src.tasks.note_tasks import add_note
from src.tasks.schedule_tasks import add_schedule, get_schedules


def seed_database(database_path: str | Path | None = None) -> None:
    create_database(database_path)
    demo_users = (("demo-anh", "Anh"), ("demo-loc", "Lộc"))
    for user_id, name in demo_users:
        if not get_user(user_id, database_path):
            create_user(user_id, name, f"models/application/user_embeddings/{user_id}.npy", database_path)

    if not get_schedules("demo-anh", database_path=database_path):
        add_schedule("demo-anh", "Học Thống kê", "2026-08-04", "08:00", "Phòng B201", database_path)
    if not get_schedules("demo-loc", database_path=database_path):
        add_schedule("demo-loc", "Học Máy", "2026-08-05", "13:00", None, database_path)
    from src.tasks.note_tasks import get_private_notes
    if not get_private_notes("demo-anh", database_path):
        add_note("demo-anh", "Hoàn thành báo cáo trước thứ Sáu.", True, database_path)


if __name__ == "__main__":
    seed_database()
    print("Demo database ready.")
