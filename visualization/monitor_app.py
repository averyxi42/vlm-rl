# monitor_app.py
from flask import Flask, render_template, send_file
import logging
import os # Import the os module

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/render')
def render():
    """Serves the rendered image, now with a check to ensure the file exists."""
    # Check if the image file exists before trying to send it.
    if not os.path.exists('render.png'):
        # If it doesn't exist, return a "Not Found" error.
        # This is much cleaner than crashing and tells the browser what's happening.
        return "render.png not found. Waiting for first callback.", 404

    # If the file exists, send it and add headers to prevent caching.
    response = send_file('render.png', mimetype='image/png')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def run_app():
    """Runs the Flask web server."""
    print("--- Flask web server thread started. Attempting to run on 0.0.0.0:5000 ---")
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)