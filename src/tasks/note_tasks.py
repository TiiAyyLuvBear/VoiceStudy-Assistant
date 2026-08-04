"""Private note queries scoped to note owner."""

from __future__ import annotations

from pathlib import Path

from src.database.database import create_database, get_connection


def add_note(user_id: str, content: str, is_private: bool = True, database_path: str | Path | None = None) -> dict:
    create_database(database_path)
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO notes (user_id, content, is_private) VALUES (?, ?, ?)",
            (user_id, content, int(is_private)),
        )
        row = connection.execute("SELECT * FROM notes WHERE note_id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def get_private_notes(user_id: str, database_path: str | Path | None = None) -> list[dict]:
    create_database(database_path)
    with get_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM notes WHERE user_id = ? AND is_private = 1 ORDER BY note_id DESC", (user_id,)
        )]
