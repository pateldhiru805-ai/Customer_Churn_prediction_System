"""Database abstraction and integration layer for Customer Churn Prediction.

Connects to MySQL with automatic reconnection, parameterized SQL queries,
and a seamless SQLite fallback when MySQL is unavailable or offline.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pymysql
from pymysql.cursors import DictCursor

try:
    from src.config import (
        DB_HOST,
        DB_NAME,
        DB_PASSWORD,
        DB_PORT,
        DB_USER,
        SQLITE_DB_PATH,
    )
except ImportError:
    from config import (
        DB_HOST,
        DB_NAME,
        DB_PASSWORD,
        DB_PORT,
        DB_USER,
        SQLITE_DB_PATH,
    )

logger = logging.getLogger("churn_db")
logging.basicConfig(level=logging.INFO)


def _connect_mysql():
    """Attempt connecting to MySQL database server."""
    # First ensure the database exists
    conn_root = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
        connect_timeout=3,
        cursorclass=DictCursor,
    )
    with conn_root.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
    conn_root.close()

    # Now connect to the database
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True,
        connect_timeout=3,
        cursorclass=DictCursor,
    )


import time

_mysql_disabled_until = 0.0


def _connect_sqlite():
    """Fallback connection to local SQLite database."""
    conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db_connection() -> Tuple[Any, str]:
    """Get an active database connection and its type ('mysql' or 'sqlite')."""
    global _mysql_disabled_until

    # Check if SQLite is explicitly requested
    db_type_pref = os.getenv("DB_TYPE", "auto").strip().lower()
    if db_type_pref == "sqlite":
        conn = _connect_sqlite()
        return conn, "sqlite"

    now = time.time()
    if now < _mysql_disabled_until:
        conn = _connect_sqlite()
        return conn, "sqlite"

    try:
        conn = _connect_mysql()
        return conn, "mysql"
    except Exception as mysql_err:
        _mysql_disabled_until = now + 300.0  # Cooldown 5 minutes before retrying MySQL
        err_msg = mysql_err.args[1] if hasattr(mysql_err, "args") and len(mysql_err.args) > 1 else str(mysql_err)
        logger.info(
            f"MySQL not connected ({err_msg}). Operating with persistent SQLite database: {SQLITE_DB_PATH.name}"
        )
        try:
            conn = _connect_sqlite()
            return conn, "sqlite"
        except Exception as sqlite_err:
            logger.error(f"SQLite connection failed: {sqlite_err}")
            raise


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    conn = None
    try:
        conn, db_type = get_db_connection()
        if db_type == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        customer_id VARCHAR(50) DEFAULT NULL,
                        input_json JSON NOT NULL,
                        tenure INT NOT NULL,
                        contract VARCHAR(50) NOT NULL,
                        monthly_charges DECIMAL(8,2) NOT NULL,
                        total_charges DECIMAL(10,2) NOT NULL,
                        churn_probability DECIMAL(5,4) NOT NULL,
                        prediction_label VARCHAR(30) NOT NULL,
                        risk_level VARCHAR(20) NOT NULL,
                        model_name VARCHAR(50) NOT NULL,
                        model_version VARCHAR(20) NOT NULL,
                        notes TEXT DEFAULT NULL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_runs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_name VARCHAR(50) NOT NULL,
                        model_version VARCHAR(20) NOT NULL,
                        roc_auc DECIMAL(5,4) NOT NULL,
                        pr_auc DECIMAL(5,4) NOT NULL,
                        f1_churn DECIMAL(5,4) NOT NULL,
                        recall_churn DECIMAL(5,4) NOT NULL,
                        precision_churn DECIMAL(5,4) NOT NULL,
                        accuracy DECIMAL(5,4) NOT NULL,
                        threshold DECIMAL(4,3) NOT NULL,
                        hyperparameters JSON DEFAULT NULL,
                        notes TEXT DEFAULT NULL
                    )
                """)
        else:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        customer_id TEXT DEFAULT NULL,
                        input_json TEXT NOT NULL,
                        tenure INTEGER NOT NULL,
                        contract TEXT NOT NULL,
                        monthly_charges REAL NOT NULL,
                        total_charges REAL NOT NULL,
                        churn_probability REAL NOT NULL,
                        prediction_label TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        notes TEXT DEFAULT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        roc_auc REAL NOT NULL,
                        pr_auc REAL NOT NULL,
                        f1_churn REAL NOT NULL,
                        recall_churn REAL NOT NULL,
                        precision_churn REAL NOT NULL,
                        accuracy REAL NOT NULL,
                        threshold REAL NOT NULL,
                        hyperparameters TEXT DEFAULT NULL,
                        notes TEXT DEFAULT NULL
                    )
                """)
        logger.info(f"Database initialized successfully ({db_type}).")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()


def save_prediction(
    input_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    customer_id: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Save an inference prediction record to the database using parameterized query.

    Returns:
        (saved: bool, error_message: Optional[str])
    """
    conn = None
    try:
        conn, db_type = get_db_connection()
        input_json_str = json.dumps(input_data)
        tenure = int(input_data.get("tenure", 0))
        contract = str(input_data.get("Contract", "Month-to-month"))
        monthly_charges = float(input_data.get("MonthlyCharges", 0.0))
        total_charges = float(input_data.get("TotalCharges", 0.0))
        prob = float(prediction_result.get("churn_probability", 0.0))
        label = str(prediction_result.get("prediction", "Unknown"))
        risk_level = str(prediction_result.get("risk_level", "Unknown"))
        model_name = str(prediction_result.get("model_name", "churn_model"))
        model_ver = str(prediction_result.get("model_version", "1.0.0"))

        if db_type == "mysql":
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO predictions (
                        customer_id, input_json, tenure, contract,
                        monthly_charges, total_charges, churn_probability,
                        prediction_label, risk_level, model_name, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(
                    sql,
                    (
                        customer_id,
                        input_json_str,
                        tenure,
                        contract,
                        monthly_charges,
                        total_charges,
                        prob,
                        label,
                        risk_level,
                        model_name,
                        model_ver,
                    ),
                )
        else:
            with conn:
                sql = """
                    INSERT INTO predictions (
                        customer_id, input_json, tenure, contract,
                        monthly_charges, total_charges, churn_probability,
                        prediction_label, risk_level, model_name, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute(
                    sql,
                    (
                        customer_id,
                        input_json_str,
                        tenure,
                        contract,
                        monthly_charges,
                        total_charges,
                        prob,
                        label,
                        risk_level,
                        model_name,
                        model_ver,
                    ),
                )
        return True, None
    except Exception as e:
        logger.error(f"Failed to save prediction record: {e}")
        return False, str(e)
    finally:
        if conn:
            conn.close()


def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve the most recent predictions ordered chronologically descending."""
    conn = None
    try:
        conn, db_type = get_db_connection()
        if db_type == "mysql":
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, created_at, customer_id, tenure, contract,
                           monthly_charges, total_charges, churn_probability,
                           prediction_label, risk_level, model_name, model_version, input_json
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT %s
                """,
                    (limit,),
                )
                rows = cur.fetchall()
                # Format dates and JSON
                for r in rows:
                    if isinstance(r.get("created_at"), datetime):
                        r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(r.get("input_json"), str):
                        try:
                            r["input_json"] = json.loads(r["input_json"])
                        except Exception:
                            pass
                return rows
        else:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT id, created_at, customer_id, tenure, contract,
                           monthly_charges, total_charges, churn_probability,
                           prediction_label, risk_level, model_name, model_version, input_json
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT ?
                """,
                    (limit,),
                )
                rows = [dict(r) for r in cur.fetchall()]
                for r in rows:
                    if isinstance(r.get("input_json"), str):
                        try:
                            r["input_json"] = json.loads(r["input_json"])
                        except Exception:
                            pass
                return rows
    except Exception as e:
        logger.error(f"Failed to fetch recent predictions: {e}")
        return []
    finally:
        if conn:
            conn.close()


def check_db_health() -> Dict[str, Any]:
    """Test connection and return health summary."""
    try:
        conn, db_type = get_db_connection()
        count = 0
        if db_type == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total FROM predictions")
                res = cur.fetchone()
                count = res["total"] if res else 0
        else:
            with conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) AS total FROM predictions")
                res = cur.fetchone()
                count = res[0] if res else 0
        conn.close()
        return {
            "status": "healthy",
            "connected": True,
            "engine": db_type,
            "predictions_count": count,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "connected": False,
            "error": str(e),
        }
