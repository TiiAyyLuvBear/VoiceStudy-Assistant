"""Schedule operations scoped to a single owner."""

from __future__ import annotations

from pathlib import Path

from src.database.database import create_database, get_connection


def add_schedule(user_id: str, title: str, date: str, time: str, description: str | None = None, database_path: str | Path | None = None) -> dict:
    create_database(database_path)
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO schedules (user_id, title, date, time, description) VALUES (?, ?, ?, ?, ?)",
            (user_id, title, date, time, description),
        )
        row = connection.execute("SELECT * FROM schedules WHERE schedule_id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def get_schedules(user_id: str, date: str | None = None, database_path: str | Path | None = None) -> list[dict]:
    create_database(database_path)
    query = "SELECT * FROM schedules WHERE user_id = ?"
    values: list[str] = [user_id]
    if date:
        query += " AND date = ?"
        values.append(date)
    query += " ORDER BY date, time, schedule_id"
    with get_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(query, values)]
