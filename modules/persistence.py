"""
Lightweight SQLite persistence for project routing and user preferences.

Scope:
    - project-level continuity (project_id)
    - user-level inferred preferences (user_id)

Design goals:
    - simple schema
    - easy to refactor later
    - no external server dependency
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any


DEFAULT_USER_PREFERENCES: dict[str, dict[str, Any]] = {
    "answer_verbosity": {"value": "balanced", "confidence": 0.5},
    "format_style": {"value": "steps_first", "confidence": 0.5},
    "risk_posture": {"value": "balanced", "confidence": 0.5},
    "citation_preference": {"value": "normal", "confidence": 0.5},
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _db_path() -> str:
    return os.getenv("APP_STATE_DB_PATH", ".data/app_state.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_state_db() -> None:
    """Create project + preference tables if missing."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL DEFAULT '',
                problem_description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Backward-compatible migration for existing deployments.
        project_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('projects')").fetchall()
        }
        if "model_id" not in project_columns:
            conn.execute(
                "ALTER TABLE projects ADD COLUMN model_id TEXT NOT NULL DEFAULT ''")
        if "problem_description" not in project_columns:
            conn.execute(
                "ALTER TABLE projects ADD COLUMN problem_description TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_users (
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                joined_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                PRIMARY KEY (project_id, user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_project(project_id: str) -> dict[str, Any] | None:
    """Return project metadata if it exists."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT project_id, model_id, problem_description, created_at, updated_at
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        if row is None:
            return None

        users = conn.execute(
            """
            SELECT user_id, joined_at, last_seen_at
            FROM project_users
            WHERE project_id = ?
            ORDER BY joined_at ASC
            """,
            (project_id,),
        ).fetchall()
        return {
            "project_id": row["project_id"],
            "model_id": row["model_id"],
            "problem_description": row["problem_description"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "users": [dict(u) for u in users],
        }


def list_projects_for_user(user_id: str) -> list[dict[str, Any]]:
    """Return projects associated with a given user_id."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                p.project_id,
                p.model_id,
                p.problem_description,
                p.created_at,
                p.updated_at,
                pu.last_seen_at
            FROM projects p
            INNER JOIN project_users pu ON pu.project_id = p.project_id
            WHERE pu.user_id = ?
            ORDER BY p.updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def touch_project(
    project_id: str,
    user_id: str,
    *,
    model_id: str | None = None,
    problem_description: str | None = None,
    create_if_missing: bool = True,
    allow_join_existing: bool = False,
) -> dict[str, Any]:
    """
    Upsert project and register user access.

    Behavior:
        - if missing:
            - create project when create_if_missing=True
            - return not_found when create_if_missing=False
        - if existing:
            - resume when user is already linked
            - optionally join when allow_join_existing=True
            - otherwise return forbidden
    """
    now = _now_iso()
    created = False
    resumed = False
    joined = False

    with _connect() as conn:
        row = conn.execute(
            "SELECT project_id FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            if not create_if_missing:
                return {
                    "status": "not_found",
                    "project_id": project_id,
                    "user_id": user_id,
                }
            model_value = (model_id or "").strip()
            problem_value = (problem_description or "").strip()
            conn.execute(
                """
                INSERT INTO projects(project_id, model_id, problem_description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, model_value, problem_value, now, now),
            )
            created = True
            conn.execute(
                """
                INSERT INTO project_users(project_id, user_id, joined_at, last_seen_at)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, user_id, now, now),
            )
        else:
            membership = conn.execute(
                """
                SELECT 1
                FROM project_users
                WHERE project_id = ? AND user_id = ?
                LIMIT 1
                """,
                (project_id, user_id),
            ).fetchone()
            if membership is not None:
                resumed = True
                conn.execute(
                    """
                    UPDATE projects
                    SET
                        updated_at = ?,
                        model_id = COALESCE(NULLIF(?, ''), model_id),
                        problem_description = COALESCE(NULLIF(?, ''), problem_description)
                    WHERE project_id = ?
                    """,
                    (now, (model_id or "").strip(),
                     (problem_description or "").strip(), project_id),
                )
                conn.execute(
                    """
                    UPDATE project_users
                    SET last_seen_at = ?
                    WHERE project_id = ? AND user_id = ?
                    """,
                    (now, project_id, user_id),
                )
            elif allow_join_existing:
                joined = True
                conn.execute(
                    """
                    INSERT INTO project_users(project_id, user_id, joined_at, last_seen_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (project_id, user_id, now, now),
                )
                conn.execute(
                    """
                    UPDATE projects
                    SET
                        updated_at = ?,
                        model_id = COALESCE(NULLIF(?, ''), model_id),
                        problem_description = COALESCE(NULLIF(?, ''), problem_description)
                    WHERE project_id = ?
                    """,
                    (now, (model_id or "").strip(),
                     (problem_description or "").strip(), project_id),
                )
            else:
                return {
                    "status": "forbidden",
                    "project_id": project_id,
                    "user_id": user_id,
                }
        conn.commit()

    project = get_project(project_id) or {
        "project_id": project_id, "users": []}
    project["created"] = created
    project["resumed"] = resumed
    project["joined"] = joined
    project["status"] = "created" if created else (
        "resumed" if resumed else ("joined" if joined else "ok"))
    return project


def get_user_preferences(user_id: str) -> dict[str, Any]:
    """Load preferences for user_id or create defaults."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT preferences_json FROM user_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row is None:
            prefs = {
                **DEFAULT_USER_PREFERENCES,
                "meta": {"observations": 0, "updated_at": _now_iso()},
            }
            conn.execute(
                """
                INSERT INTO user_preferences(user_id, preferences_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (user_id, json.dumps(prefs), _now_iso()),
            )
            conn.commit()
            return prefs

        return json.loads(row["preferences_json"])


def _extract_signal(user_text: str, phrases: tuple[str, ...]) -> bool:
    lower = user_text.lower()
    return any(p in lower for p in phrases)


def _update_pref_value(prefs: dict[str, Any], key: str, value: str) -> None:
    current = prefs.get(key, {"value": value, "confidence": 0.5})
    current["value"] = value
    current["confidence"] = round(
        min(1.0, float(current.get("confidence", 0.5)) + 0.12), 2)
    prefs[key] = current


def infer_and_update_user_preferences(
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> dict[str, Any]:
    """
    Infer stable style preferences from conversation cues.

    This only affects response style. It must not change factual retrieval.
    """
    prefs = get_user_preferences(user_id)

    if _extract_signal(user_message, ("short answer", "concise", "brief", "quick")):
        _update_pref_value(prefs, "answer_verbosity", "concise")
    elif _extract_signal(user_message, ("detailed", "explain", "why", "deep dive")):
        _update_pref_value(prefs, "answer_verbosity", "detailed")

    if _extract_signal(user_message, ("step by step", "steps", "procedure", "bullet")):
        _update_pref_value(prefs, "format_style", "steps_first")
    elif _extract_signal(user_message, ("summary first", "overview first", "big picture")):
        _update_pref_value(prefs, "format_style", "summary_first")

    if _extract_signal(user_message, ("source", "cite", "reference", "document section")):
        _update_pref_value(prefs, "citation_preference", "high")

    if _extract_signal(user_message, ("safety", "risk", "hazard", "compliance", "warning")):
        _update_pref_value(prefs, "risk_posture", "conservative")

    meta = prefs.get("meta", {"observations": 0})
    meta["observations"] = int(meta.get("observations", 0)) + 1
    meta["updated_at"] = _now_iso()
    meta["last_assistant_chars"] = len(assistant_message)
    prefs["meta"] = meta

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_preferences(user_id, preferences_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET preferences_json = excluded.preferences_json, updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(prefs), _now_iso()),
        )
        conn.commit()

    return prefs


def _checkpoint_db_path() -> str:
    return os.getenv("LANGGRAPH_CHECKPOINT_DB", ".data/checkpoints.db")


def _delete_thread_from_checkpoint_db(project_id: str) -> dict[str, int]:
    """
    Best-effort cleanup of LangGraph checkpoint rows for one thread_id.

    We introspect tables and delete rows from any table that has a thread_id
    column, so this stays resilient across checkpoint schema versions.
    """
    path = _checkpoint_db_path()
    if not os.path.exists(path):
        return {"tables_touched": 0, "rows_deleted": 0}

    tables_touched = 0
    rows_deleted = 0
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()

        for table in tables:
            table_name = table["name"]
            cols = conn.execute(
                f"PRAGMA table_info('{table_name}')").fetchall()
            col_names = {c["name"] for c in cols}
            if "thread_id" not in col_names:
                continue

            cur = conn.execute(
                f"DELETE FROM '{table_name}' WHERE thread_id = ?",
                (project_id,),
            )
            tables_touched += 1
            rows_deleted += int(cur.rowcount if cur.rowcount != -1 else 0)

        conn.commit()

    return {"tables_touched": tables_touched, "rows_deleted": rows_deleted}


def delete_project(project_id: str, user_id: str) -> dict[str, Any]:
    """
    Delete a project for a user.

    Rules:
      - project must exist
      - user_id must already be associated with project
      - removes project metadata + user associations
      - best-effort removes checkpoint history for thread_id == project_id
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT project_id FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return {"status": "not_found", "project_id": project_id}

        membership = conn.execute(
            """
            SELECT 1
            FROM project_users
            WHERE project_id = ? AND user_id = ?
            LIMIT 1
            """,
            (project_id, user_id),
        ).fetchone()
        if membership is None:
            return {"status": "forbidden", "project_id": project_id, "user_id": user_id}

        conn.execute(
            "DELETE FROM project_users WHERE project_id = ?", (project_id,))
        conn.execute(
            "DELETE FROM projects WHERE project_id = ?", (project_id,))
        conn.commit()

    checkpoint_cleanup = _delete_thread_from_checkpoint_db(project_id)
    return {
        "status": "deleted",
        "project_id": project_id,
        "checkpoint_cleanup": checkpoint_cleanup,
    }
