from flask import Flask, request, send_file, render_template, session, jsonify, redirect
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import sqlite3
import os
from qec_project import api_run

app = Flask(__name__, template_folder=".")
app.secret_key = "super_secret_quantum_key" # Required for Flask sessions
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def init_db():
    """Initializes the database table using standard SQL commands."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # 1. Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # 2. Experiment History Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            noise_type TEXT,
            p_value REAL,
            episodes INTEGER,
            before_rate REAL,
            after_rate REAL,
            avg_reward REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create a default admin user if the table is empty
    c.execute("SELECT * FROM users WHERE username='admin'")
    if c.fetchone() is None:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                  ('admin', generate_password_hash('password123')))
    
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Using a parameterized SELECT clause to prevent SQL injection
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()

    if row and check_password_hash(row[0], password):
        session['logged_in'] = True
        session['username'] = username
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return jsonify({"success": True})

@app.route("/")
def home():
    # If not logged in, we let the frontend know so it shows the login screen
    if not session.get('logged_in'):
        return render_template("index.html", logged_in=False)
    return render_template("index.html", logged_in=True, username=session.get('username'))

@app.route("/download-rl")
def download_rl():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    noise_type = request.args.get("noise_type", "bitflip")
    fname = f"rl_training_dataset_{noise_type}.csv"
    return send_file(fname, as_attachment=True)

@app.route("/get-history")
def get_history():
    """Fetches the past 10 experiments for the logged-in user."""
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    username = session.get('username')
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row # This allows us to convert rows to dictionaries easily
    c = conn.cursor()
    c.execute("SELECT * FROM runs WHERE username=? ORDER BY timestamp DESC LIMIT 10", (username,))
    rows = c.fetchall()
    conn.close()
    
    # Convert SQLite rows to a list of Python dictionaries
    history = [dict(row) for row in rows]
    return jsonify(history)
def run_rl_background(data, sid, username):
    noise_type = data.get("noise_type", "bitflip")
    p = float(data.get("p", 0.05))
    shots = min(int(data.get("shots", 2000)), 800)
    episodes = min(int(data.get("episodes", 20000)), 1500)
    trials = min(int(data.get("trials", 5000)), 1000)

    def send_progress(percent):
        socketio.emit('progress', {'progress': percent}, to=sid)

    # Run the heavy simulation
    result_data = api_run(
        noise_type=noise_type,
        p=p,
        shots=shots,
        episodes=episodes,
        trials=trials,
        seed=42,
        progress_callback=send_progress
    )

    # Save the results to the database!
    if username:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''INSERT INTO runs 
                     (username, noise_type, p_value, episodes, before_rate, after_rate, avg_reward) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (username, noise_type, p, episodes, 
                   result_data['before_rate'], result_data['after_rate'], result_data['avg_reward']))
        conn.commit()
        conn.close()

    # Emit the final result back to the specific user
    socketio.emit('result', {'result': result_data}, to=sid)


@socketio.on('start_run')
def handle_run_project(data):
    user_sid = request.sid 
    # Grab the username from the Flask session so we know who ran this
    username = session.get('username') 
    
    thread = threading.Thread(target=run_rl_background, args=(data, user_sid, username))
    thread.start()

if __name__ == "__main__":
    socketio.run(app, port=5000, debug=True)