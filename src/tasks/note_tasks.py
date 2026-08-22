"""Private note queries scoped to note owner."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from src.database.database import create_database, get_connection


def add_note(user_id: str, content: str, is_private: bool = True, database_path: str | Path | None = None) -> dict:
    create_database(database_path)
    with closing(get_connection(database_path)) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO notes (user_id, content, is_private) VALUES (?, ?, ?)",
                (user_id, content, int(is_private)),
            )
            row = connection.execute("SELECT * FROM notes WHERE note_id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def get_private_notes(user_id: str, database_path: str | Path | None = None) -> list[dict]:
    create_database(database_path)
    with closing(get_connection(database_path)) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM notes WHERE user_id = ? AND is_private = 1 ORDER BY note_id DESC", (user_id,)
        )]


def get_notes(user_id: str, database_path: str | Path | None = None) -> list[dict]:
    create_database(database_path)
    with closing(get_connection(database_path)) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM notes WHERE user_id = ? ORDER BY note_id DESC", (user_id,)
        )]


def delete_note(user_id: str, note_id: int, database_path: str | Path | None = None) -> bool:
    create_database(database_path)
    with closing(get_connection(database_path)) as connection:
        with connection:
            result = connection.execute(
                "DELETE FROM notes WHERE user_id = ? AND note_id = ?",
                (user_id, int(note_id)),
            )
    return result.rowcount == 1
