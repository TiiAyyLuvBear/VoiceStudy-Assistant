# Viết hàm tạo database và bảng nếu chưa tồn tại (SQLite)

import sqlite3
from contextlib import closing
from pathlib import Path

from src.utils.config import load_yaml_mapping

CONFIG_PATH = "config.yaml"
config, _ = load_yaml_mapping(CONFIG_PATH)
DATABASE_PATH = config.get('database', {}).get('path', 'voicestudy.db')
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

def _resolve_database_path(database_path: str | Path | None) -> str | Path:
    """Use configured database when caller does not provide a path."""
    return DATABASE_PATH if database_path is None else database_path


def get_connection(database_path: str | Path | None = None) -> sqlite3.Connection:
    resolved = _resolve_database_path(database_path)
    if str(resolved) != ":memory:":
        Path(resolved).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_database(database_path: str | Path | None = None) -> None:
    with closing(get_connection(database_path)) as connection:
        with connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            _ensure_user_secret_columns(connection)


def _ensure_user_secret_columns(connection: sqlite3.Connection) -> None:
    """Add enrollment-secret columns to databases created by older versions."""

    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(users)").fetchall()
    }
    migrations = {
        "secret_phrase_hash": "ALTER TABLE users ADD COLUMN secret_phrase_hash TEXT",
        "secret_phrase_salt": "ALTER TABLE users ADD COLUMN secret_phrase_salt TEXT",
        "secret_phrase_updated_at": (
            "ALTER TABLE users ADD COLUMN secret_phrase_updated_at TEXT"
        ),
    }
    for column, statement in migrations.items():
        if column not in columns:
            connection.execute(statement)

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
