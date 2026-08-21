import json
import os
import csv
import io
import hashlib
import secrets
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template_string, request, redirect,
    url_for, session, flash, Response, jsonify,
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

OUTPUT_DIR = "output"
SECURE_MAP_FILE = os.path.join(OUTPUT_DIR, "secure_mapping.json")
SECURED_DATA_FILE = os.path.join(OUTPUT_DIR, "secured_data.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "detokenization_audit.log")

AUTHORIZED_USERS = {
    "admin": hashlib.sha256("Admin".encode()).hexdigest(),
    "auditor": hashlib.sha256("Audit".encode()).hexdigest(),
    "manager": hashlib.sha256("Manager".encode()).hexdigest(),
}

USER_ROLES = {
    "admin": "Administrator",
    "auditor": "Auditor",
    "manager": "Branch Manager",
}


def write_log(username, action, detail="", reason=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ip = request.remote_addr if request else "N/A"
    entry = f"[{timestamp}] | USER: {username} | ROLE: {USER_ROLES.get(username, 'Unknown')} | IP: {ip} | ACTION: {action} | DETAIL: {detail} | REASON: {reason}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def load_secure_mapping():
    if not os.path.exists(SECURE_MAP_FILE):
        return None
    with open(SECURE_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hash_map():
    mapping = load_secure_mapping()
    return mapping.get("hash_map") if mapping else None


def load_row_map():
    mapping = load_secure_mapping()
    if not mapping:
        return {}, {}
    return mapping.get("row_pii", {}), mapping.get("token_to_row", {})


def reconstruct_full_row(token_value, token_map):
    row_pii, token_to_row = load_row_map()
    hash_map = load_hash_map()
    if not row_pii or not token_to_row or not hash_map:
        return None

    id_hash = token_to_row.get(token_value)
    if not id_hash:
        return None

    pii_tokens = row_pii.get(id_hash, {})

    restored_pii = {}
    for pii_col, tok in pii_tokens.items():
        if pii_col in token_map:
            reverse = {v: k for k, v in token_map[pii_col].items()}
            restored_pii[pii_col] = reverse.get(tok, tok)
        else:
            restored_pii[pii_col] = tok

    restored_ids = {}
    for id_col, col_map in hash_map.items():
        reverse_hash = {v: k for k, v in col_map.items()}
        if id_hash in reverse_hash:
            restored_ids[id_col] = reverse_hash[id_hash]

    if not os.path.exists(SECURED_DATA_FILE):
        return None
    secured_df = pd.read_csv(SECURED_DATA_FILE, dtype=str, nrows=0)
    all_secured_cols = secured_df.columns.tolist()
    secured_df = pd.read_csv(SECURED_DATA_FILE, dtype=str)
    row_match = secured_df[secured_df["id_hash"] == id_hash]
    if row_match.empty:
        return None
    secured_row = row_match.iloc[0].to_dict()

    result = []
    for id_col, orig_val in restored_ids.items():
        result.append((id_col, orig_val))
    for pii_col, orig_val in restored_pii.items():
        result.append((pii_col, orig_val))
    for col in all_secured_cols:
        if col.endswith("_hash"):
            continue
        if col not in restored_pii and col not in restored_ids:
            result.append((col, secured_row.get(col, "")))

    return result


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def load_token_map():
    mapping = load_secure_mapping()
    return mapping.get("token_map") if mapping else None


TEMPLATE_BASE = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }} - DeTokenization Portal</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a14;color:#e0e0e0;min-height:100vh}
.nav{background:linear-gradient(135deg,#12122a,#1a1a3e);padding:14px 30px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #2a2a5c}
.nav-brand{font-size:18px;font-weight:700;background:linear-gradient(135deg,#6c5ce7,#00cec9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-links a{color:#b0b0cc;text-decoration:none;margin-left:20px;font-size:14px;transition:color .2s}
.nav-links a:hover{color:#6c5ce7}
.nav-user{color:#8a8aaa;font-size:13px}
.nav-user strong{color:#00cec9}
.container{max-width:960px;margin:30px auto;padding:0 20px}
.card{background:linear-gradient(135deg,#14142a,#1a1a36);border:1px solid #2a2a5c;border-radius:12px;padding:28px;margin-bottom:24px;transition:border-color .3s}
.card:hover{border-color:#6c5ce7}
.card h2{font-size:20px;margin-bottom:16px;color:#e0e0e0}
.card h2 span{color:#6c5ce7}
.form-group{margin-bottom:16px}
.form-group label{display:block;font-size:13px;color:#8a8aaa;margin-bottom:6px}
.form-group input,.form-group select,.form-group textarea{width:100%;padding:10px 14px;background:#0f0f22;border:1px solid #2a2a5c;border-radius:8px;color:#e0e0e0;font-size:14px;transition:border-color .2s}
.form-group input:focus,.form-group textarea:focus{outline:none;border-color:#6c5ce7}
.btn{display:inline-block;padding:10px 24px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;text-decoration:none}
.btn-primary{background:linear-gradient(135deg,#6c5ce7,#5a4bd6);color:#fff}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(108,92,231,.4)}
.btn-danger{background:linear-gradient(135deg,#d63031,#c02020);color:#fff}
.btn-danger:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(214,48,49,.4)}
.btn-secondary{background:#2a2a5c;color:#e0e0e0}
.btn-secondary:hover{background:#3a3a6c}
.btn-success{background:linear-gradient(135deg,#00b894,#00a382);color:#fff}
.btn-success:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(0,184,148,.4)}
.result-box{background:#0f0f22;border:1px solid #2a2a5c;border-radius:8px;padding:16px;margin-top:12px;word-break:break-all;font-family:'Cascadia Code',monospace;font-size:14px}
.result-box .label{color:#8a8aaa;font-size:12px;margin-bottom:4px}
.result-box .value{color:#00cec9;font-size:16px;font-weight:600}
.flash{padding:12px 18px;border-radius:8px;margin-bottom:16px;font-size:14px}
.flash-warning{background:rgba(253,203,110,.1);border:1px solid #fdcb6e;color:#fdcb6e}
.flash-success{background:rgba(0,184,148,.1);border:1px solid #00b894;color:#00b894}
.flash-error{background:rgba(214,48,49,.1);border:1px solid #d63031;color:#d63031}
.login-wrapper{display:flex;justify-content:center;align-items:center;min-height:80vh}
.login-card{width:400px}
table{width:100%;border-collapse:collapse;font-size:13px}
table th{background:#1a1a3e;color:#8a8aaa;padding:10px;text-align:left;border-bottom:1px solid #2a2a5c}
table td{padding:10px;border-bottom:1px solid #1a1a3e;color:#c0c0d0;word-break:break-all}
table tr:hover{background:rgba(108,92,231,.05)}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600}
.badge-login{background:rgba(0,206,201,.15);color:#00cec9}
.badge-detokenize{background:rgba(108,92,231,.15);color:#6c5ce7}
.badge-export{background:rgba(253,203,110,.15);color:#fdcb6e}
.badge-logout{background:rgba(214,48,49,.15);color:#d63031}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}
.stat-card{background:#0f0f22;border:1px solid #2a2a5c;border-radius:10px;padding:18px;text-align:center}
.stat-card .num{font-size:28px;font-weight:700;background:linear-gradient(135deg,#6c5ce7,#00cec9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stat-card .lbl{font-size:12px;color:#8a8aaa;margin-top:4px}
.actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
@media print{body{background:#fff;color:#000}.nav,.btn,.actions{display:none!important}table th{background:#eee;color:#333}table td{color:#333;border-color:#ddd}.card{border-color:#ddd;background:#fff}}
</style>
</head>
<body>
{% if session.get('username') %}
<nav class="nav">
<div class="nav-brand">DeTokenization Portal</div>
<div class="nav-links">
<a href="{{ url_for('dashboard') }}">Dashboard</a>
<a href="{{ url_for('detokenize_page') }}">Detokenize</a>
<a href="{{ url_for('logs_page') }}">Audit Logs</a>
<span class="nav-user">| <strong>{{ session.get('username') }}</strong> ({{ roles.get(session.get('username'),'') }})</span>
<a href="{{ url_for('logout') }}" style="color:#d63031">Logout</a>
</div>
</nav>
{% endif %}
<div class="container">
{% with messages = get_flashed_messages(with_categories=true) %}
{% for cat, msg in messages %}
<div class="flash flash-{{ cat }}">{{ msg }}</div>
{% endfor %}
{% endwith %}
{% block content %}{% endblock %}
</div>
</body>
</html>
"""

TEMPLATE_LOGIN = """
{% extends "base" %}
{% block content %}
<div class="login-wrapper">
<div class="card login-card">
<h2><span>&#128274;</span> Secure Login</h2>
<p style="color:#8a8aaa;font-size:13px;margin-bottom:20px">Authorized personnel only (Decree 13/2023, Article 8)</p>
<form method="POST">
<div class="form-group"><label>Username</label><input type="text" name="username" required autofocus></div>
<div class="form-group"><label>Password</label><input type="password" name="password" required></div>
<button type="submit" class="btn btn-primary" style="width:100%">Login</button>
</form>
</div>
</div>
{% endblock %}
"""

TEMPLATE_DASHBOARD = """
{% extends "base" %}
{% block content %}
<h2 style="margin-bottom:20px">Dashboard</h2>
<div class="stats">
<div class="stat-card"><div class="num">{{ total_tokens }}</div><div class="lbl">Total Tokens</div></div>
<div class="stat-card"><div class="num">{{ total_columns }}</div><div class="lbl">PII Columns</div></div>
<div class="stat-card"><div class="num">{{ log_count }}</div><div class="lbl">Audit Entries</div></div>
</div>
<div class="card">
<h2><span>&#128220;</span> Token Map Summary</h2>
<table>
<thead><tr><th>Column</th><th>Token Count</th><th>Sample Token</th></tr></thead>
<tbody>
{% for col, count, sample in columns_info %}
<tr><td>{{ col }}</td><td>{{ count }}</td><td style="font-family:monospace;color:#6c5ce7">{{ sample }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
{% endblock %}
"""

TEMPLATE_DETOKENIZE = """
{% extends "base" %}
{% block content %}
<div class="card">
<h2><span>&#128275;</span> Detokenize Value</h2>
<form method="POST">
<div class="form-group"><label>Token Value</label><input type="text" name="token_value" placeholder="TOKEN_xxxxxxxxxxxx" value="{{ request.form.get('token_value','') }}" required></div>
<div class="form-group"><label>Reason for access</label><input type="text" name="reason" placeholder="Audit request #..." required></div>
<button type="submit" class="btn btn-primary">Detokenize</button>
</form>
{% if matched_col %}
<div class="result-box" style="margin-top:20px">
<div class="label">Token found in column: <strong style="color:#6c5ce7">{{ matched_col }}</strong></div>
<div class="value" style="margin-bottom:12px">{{ matched_value }}</div>
</div>
{% endif %}
{% if full_row %}
<div style="margin-top:16px;overflow-x:auto">
<h3 style="font-size:16px;margin-bottom:10px;color:#00cec9">Full Customer Record</h3>
<table>
<thead><tr>{% for col in full_row_cols %}<th>{{ col }}</th>{% endfor %}</tr></thead>
<tbody><tr>{% for val in full_row_vals %}<td>{{ val }}</td>{% endfor %}</tr></tbody>
</table>
</div>
<div style="margin-top:12px">
<table style="max-width:500px">
<thead><tr><th>Column</th><th>Value</th></tr></thead>
<tbody>
{% for col, val in full_row %}
<tr><td style="color:#8a8aaa;font-weight:600">{{ col }}</td><td style="color:#00cec9">{{ val }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
{% endif %}
{% if not_found %}
<div class="result-box" style="margin-top:20px">
<div class="label">Result</div>
<div class="value" style="color:#d63031">[NOT FOUND] Token does not exist in token_map</div>
</div>
{% endif %}
</div>
{% endblock %}
"""

TEMPLATE_LOGS = """
{% extends "base" %}
{% block content %}
<div class="card">
<h2><span>&#128203;</span> Audit Logs</h2>
<div class="actions">
<a href="{{ url_for('export_logs', fmt='csv') }}" class="btn btn-success">Export CSV</a>
<button onclick="window.print()" class="btn btn-secondary">Print</button>
</div>
<div style="margin-top:20px;max-height:500px;overflow-y:auto">
<table>
<thead><tr><th>Timestamp</th><th>User</th><th>Role</th><th>IP</th><th>Action</th><th>Detail</th><th>Reason</th></tr></thead>
<tbody>
{% for entry in logs %}
<tr>
<td>{{ entry.timestamp }}</td>
<td>{{ entry.user }}</td>
<td>{{ entry.role }}</td>
<td>{{ entry.ip }}</td>
<td><span class="badge badge-{{ entry.action_class }}">{{ entry.action }}</span></td>
<td style="font-size:12px">{{ entry.detail }}</td>
<td style="font-size:12px;color:#fdcb6e">{{ entry.reason }}</td>
</tr>
{% endfor %}
{% if not logs %}
<tr><td colspan="7" style="text-align:center;color:#8a8aaa">No audit entries yet.</td></tr>
{% endif %}
</tbody>
</table>
</div>
</div>
{% endblock %}
"""

TEMPLATES = {
    "base": TEMPLATE_BASE,
    "login": TEMPLATE_LOGIN,
    "dashboard": TEMPLATE_DASHBOARD,
    "detokenize": TEMPLATE_DETOKENIZE,
    "logs": TEMPLATE_LOGS,
}


from jinja2 import BaseLoader, TemplateNotFound

class DictLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates
    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)

app.jinja_loader = DictLoader(TEMPLATES)


@app.context_processor
def inject_roles():
    return {"roles": USER_ROLES}


def parse_log_entries():
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = {}
            try:
                segments = line.split(" | ")
                parts["timestamp"] = segments[0].strip("[]")
                for seg in segments[1:]:
                    if ": " in seg:
                        k, v = seg.split(": ", 1)
                        parts[k.strip()] = v.strip()
            except Exception:
                continue
            action = parts.get("ACTION", "")
            action_class = "login"
            if "DETOKENIZE" in action.upper():
                action_class = "detokenize"
            elif "EXPORT" in action.upper():
                action_class = "export"
            elif "LOGOUT" in action.upper():
                action_class = "logout"
            detail_raw = parts.get("DETAIL", "")
            reason_raw = parts.get("REASON", "")
            if not reason_raw and "Reason: " in detail_raw:
                detail_parts = detail_raw.split(" | Reason: ", 1)
                detail_clean = detail_parts[0]
                reason_raw = detail_parts[1] if len(detail_parts) > 1 else ""
            else:
                detail_clean = detail_raw
            entries.append({
                "timestamp": parts.get("timestamp", ""),
                "user": parts.get("USER", ""),
                "role": parts.get("ROLE", ""),
                "ip": parts.get("IP", ""),
                "action": action,
                "detail": detail_clean,
                "reason": reason_raw,
                "action_class": action_class,
            })
    return list(reversed(entries))


@app.route("/", methods=["GET"])
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        pw_hash = hashlib.sha256(password.encode()).hexdigest()

        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == pw_hash:
            session["username"] = username
            write_log(username, "LOGIN", "Successful login")
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))
        else:
            write_log(username or "unknown", "LOGIN_FAILED", "Invalid credentials")
            flash("Invalid credentials.", "error")

    return render_template_string(
        "{% extends 'base' %}" + TEMPLATE_LOGIN.split("{% extends \"base\" %}")[1],
        title="Login"
    )


@app.route("/dashboard")
@login_required
def dashboard():
    token_map = load_token_map()
    total_tokens = 0
    total_columns = 0
    columns_info = []

    if token_map:
        total_columns = len(token_map)
        for col, col_map in token_map.items():
            count = len(col_map)
            total_tokens += count
            sample = list(col_map.values())[0] if col_map else "N/A"
            columns_info.append((col, count, sample))

    log_count = len(parse_log_entries())

    return render_template_string(
        "{% extends 'base' %}" + TEMPLATE_DASHBOARD.split("{% extends \"base\" %}")[1],
        title="Dashboard",
        total_tokens=total_tokens,
        total_columns=total_columns,
        log_count=log_count,
        columns_info=columns_info,
    )


@app.route("/detokenize", methods=["GET", "POST"])
@login_required
def detokenize_page():
    matched_col = None
    matched_value = None
    full_row = None
    full_row_cols = None
    full_row_vals = None
    not_found = False

    if request.method == "POST":
        token_value = request.form.get("token_value", "").strip()
        reason = request.form.get("reason", "").strip()
        token_map = load_token_map()

        if not token_map:
            flash("Token map not found.", "error")
        elif not reason:
            flash("Reason is required.", "warning")
        else:
            found = False
            for col, col_map in token_map.items():
                reverse = {v: k for k, v in col_map.items()}
                if token_value in reverse:
                    matched_value = reverse[token_value]
                    matched_col = col
                    found = True

                    row_data = reconstruct_full_row(token_value, token_map)
                    if row_data:
                        full_row = row_data
                        full_row_cols = [r[0] for r in row_data]
                        full_row_vals = [r[1] for r in row_data]

                    write_log(
                        session["username"],
                        "DETOKENIZE",
                        f"Token: {token_value} | Column: {col} | Value: {matched_value}",
                        reason=reason,
                    )
                    break

            if not found:
                not_found = True
                write_log(
                    session["username"],
                    "DETOKENIZE_NOT_FOUND",
                    f"Token: {token_value}",
                    reason=reason,
                )

    return render_template_string(
        "{% extends 'base' %}" + TEMPLATE_DETOKENIZE.split("{% extends \"base\" %}")[1],
        title="Detokenize",
        matched_col=matched_col,
        matched_value=matched_value,
        full_row=full_row,
        full_row_cols=full_row_cols,
        full_row_vals=full_row_vals,
        not_found=not_found,
    )


@app.route("/logs")
@login_required
def logs_page():
    logs = parse_log_entries()
    return render_template_string(
        "{% extends 'base' %}" + TEMPLATE_LOGS.split("{% extends \"base\" %}")[1],
        title="Audit Logs",
        logs=logs,
    )


@app.route("/export-logs/<fmt>")
@login_required
def export_logs(fmt):
    if fmt != "csv":
        return "Unsupported format", 400

    write_log(session["username"], "EXPORT_LOGS", f"Format: {fmt}")
    entries = parse_log_entries()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "User", "Role", "IP", "Action", "Detail", "Reason"])
    for e in entries:
        writer.writerow([e["timestamp"], e["user"], e["role"], e["ip"], e["action"], e["detail"], e["reason"]])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"},
    )


@app.route("/logout")
def logout():
    username = session.get("username", "unknown")
    write_log(username, "LOGOUT", "User logged out")
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 50)
    print("  DETOKENIZATION PORTAL")
    print("  Authorized access only (Decree 13/2023)")
    print("=" * 50)
    print("\n  Default accounts:")
    print("    admin    / Admin@2025!")
    print("    auditor  / Audit@2025!")
    print("    manager  / Manager@2025!")
    print(f"\n  URL: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=False, host="127.0.0.1", port=5000)
