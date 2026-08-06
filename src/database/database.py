# Viết hàm tạo database và bảng nếu chưa tồn tại (SQLite)

import sqlite3
from pathlib import Path

from src.utils.config import load_yaml_mapping

CONFIG_PATH = "config.yaml"
config, _ = load_yaml_mapping(CONFIG_PATH)
DATABASE_PATH = config.get('database', {}).get('path', 'voicestudy.db')

def _resolve_database_path(database_path: str | Path | None) -> str | Path:
    """Use configured database when caller does not provide a path."""
    return DATABASE_PATH if database_path is None else database_path


def get_connection(database_path: str | Path | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(_resolve_database_path(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_database(database_path: str | Path | None = None) -> None:
    conn = get_connection(database_path)
    cursor = conn.cursor()

    # Tạo bảng users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            embedding_path TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tạo bảng schedules
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedules (
            schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    ''')

    # Tạo bảng notes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            is_private INTEGER NOT NULL DEFAULT 1 CHECK (is_private IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()

def test_database_connection(database_path: str | Path | None = None) -> bool:
    try:
        conn = get_connection(database_path)
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return False
def __main__():
    create_database()
    if test_database_connection():
        print("Database connection successful.")
    else:
        print("Database connection failed.")
if __name__ == "__main__":
    __main__()
