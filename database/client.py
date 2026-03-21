"""
Database client initialisation (Turso/libSQL).

Uses the Turso HTTP API directly (/v2/pipeline) to avoid WebSocket
connection issues with libsql_client's wss:// transport.
"""

import json
import urllib.request
import ssl
import streamlit as st


def _parse_value(cell):
    """Convert Turso typed cell dict → plain Python value."""
    if cell is None:
        return None
    t = cell.get("type")
    v = cell.get("value")
    if t == "null" or v is None:
        return None
    if t == "integer":
        return int(v)
    if t in ("float", "real"):
        return float(v)
    return v  # text, blob → str


class ExecuteResult:
    """Wraps a Turso execute result to expose a .rows list of plain Python tuples."""
    def __init__(self, result: dict):
        self.rows = [
            [_parse_value(cell) for cell in row]
            for row in result.get("rows", [])
        ]
        self.cols = [c.get("name") for c in result.get("cols", [])]


class TursoClient:
    """Minimal Turso HTTP API client matching the libsql_client interface."""

    def __init__(self, url: str, auth_token: str):
        # Convert libsql:// → https://
        self.endpoint = url.replace("libsql://", "https://") + "/v2/pipeline"
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        self._ssl_ctx = ssl.create_default_context()

    def _pipeline(self, requests: list) -> list:
        body = json.dumps({"requests": requests}).encode()
        req = urllib.request.Request(self.endpoint, data=body, headers=self.headers)
        resp = urllib.request.urlopen(req, context=self._ssl_ctx, timeout=15)
        return json.loads(resp.read())["results"]

    def execute(self, sql: str, params: tuple = ()):
        """Execute a single statement, returns ExecuteResult with .rows list."""
        args = [{"type": "text", "value": str(p)} for p in params]
        results = self._pipeline([
            {"type": "execute", "stmt": {"sql": sql, "args": args}},
            {"type": "close"},
        ])
        raw = results[0]
        if raw.get("type") == "error":
            raise Exception(raw.get("error", {}).get("message", "DB error"))
        return ExecuteResult(raw["response"]["result"])

    def batch(self, statements):
        """
        Execute multiple statements in one round-trip.

        Accepts either:
          - list of SQL strings
          - list of (sql, params) tuples
        """
        requests = []
        for stmt in statements:
            if isinstance(stmt, tuple):
                sql, params = stmt
                args = [{"type": "text", "value": str(p)} for p in params]
            else:
                sql, args = stmt, []
            requests.append({"type": "execute", "stmt": {"sql": sql, "args": args}})
        requests.append({"type": "close"})
        return self._pipeline(requests)

    def close(self):
        pass  # HTTP is stateless — nothing to close


@st.cache_resource
def init_db():
    """Initializes the database connection using Turso HTTP API."""
    try:
        url = st.secrets["TURSO_DATABASE_URL"]
        auth_token = st.secrets["TURSO_AUTH_TOKEN"]
        client = TursoClient(url=url, auth_token=auth_token)

        client.batch([
            "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, machine_id TEXT, rpm TEXT)",
            "CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, anomaly_count INTEGER, threshold REAL, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY, analysis_id INTEGER, label_text TEXT, FOREIGN KEY (analysis_id) REFERENCES analyses (id))",
            "CREATE TABLE IF NOT EXISTS valve_events (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, crank_angle REAL, data_value REAL, curve_type TEXT, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS anomaly_tags (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, crank_angle REAL, fault_classification TEXT, tag_type TEXT DEFAULT 'Manual Tag', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS waveform_data (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, crank_angle REAL, data_value REAL, curve_type TEXT, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS configs (machine_id TEXT PRIMARY KEY, contamination REAL DEFAULT 0.05, pressure_anom_limit INT DEFAULT 10, valve_anom_limit INT DEFAULT 5, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, machine_id TEXT, cylinder TEXT, severity TEXT, message TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
        ])

        return client
    except KeyError:
        st.error("Database secrets (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Turso database: {e}")
        st.stop()
