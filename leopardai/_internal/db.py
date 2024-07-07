"""
Database for storing photon information. This is a system component - do not directly
import this module or operate on the database with direct SQL queries, unless you know
exactly what you are doing. Use the leopard CLI instead. Arbitrary changes to the database
may cause the metadata to be corrupted.
"""

import sqlite3

from ..config import DB_PATH
from ..util import create_cached_dir_if_needed

_leopard_internal_db = None


def DB() -> sqlite3.Connection:
    """
    Returns a sqlite3 connection to the database.
    """
    global _leopard_internal_db
    if not _leopard_internal_db:
        create_cached_dir_if_needed()
        _leopard_internal_db = sqlite3.connect(DB_PATH)
        _leopard_internal_db.cursor().execute(
            "CREATE TABLE IF NOT EXISTS photon (id TEXT, name TEXT, model TEXT, path"
            " TEXT, creation_time INT)"
        )
        _leopard_internal_db.commit()
    return _leopard_internal_db