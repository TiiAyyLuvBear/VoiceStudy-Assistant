"""Persistence operations for application users."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.database.database import create_database, get_connection


def _row_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


def create_user(user_id: str, name: str, embedding_path: str | None = None, database_path: str | Path | None = None) -> dict:
    create_database(database_path)
    with get_connection(database_path) as connection:
        connection.execute(
            "INSERT INTO users (user_id, name, embedding_path) VALUES (?, ?, ?)",
            (user_id, name, embedding_path),
        )
    return get_user(user_id, database_path)


def get_user(user_id: str, database_path: str | Path | None = None) -> dict | None:
    create_database(database_path)
    with get_connection(database_path) as connection:
        return _row_dict(connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())


def list_users(database_path: str | Path | None = None) -> list[dict]:
    create_database(database_path)
    with get_connection(database_path) as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM users ORDER BY name")]


def update_embedding_path(user_id: str, embedding_path: str | None, database_path: str | Path | None = None) -> bool:
    create_database(database_path)
    with get_connection(database_path) as connection:
        result = connection.execute("UPDATE users SET embedding_path = ? WHERE user_id = ?", (embedding_path, user_id))
    return result.rowcount == 1


def delete_user(user_id: str, database_path: str | Path | None = None) -> bool:
    create_database(database_path)
    with get_connection(database_path) as connection:
        result = connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    return result.rowcount == 1
