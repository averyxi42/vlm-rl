import argparse
import base64
import io
import os
import time
from threading import Thread

from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ==============================================================================
#  1. FILE WATCHER LOGIC
# ==============================================================================

# Global variable to hold our SocketIO server instance, allowing the
# file watcher thread to access it.
socketio_server = None

class ImageChangeHandler(FileSystemEventHandler):
    """
    Handles file system events. When a file is modified, it reads,
    compresses, and emits the image via WebSocket.
    """
    def __init__(self, image_path):
        self.image_path = image_path
        # A simple debounce mechanism to handle rapid-fire write events from the OS
        self.last_event_time = 0

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        # Check if the modified file is the one we are watching
        if event.src_path == self.image_path:
            # Debounce: only process if more than a short time has passed
            current_time = time.time()
            if current_time - self.last_event_time < 0.05:
                return
            self.last_event_time = current_time

            try:
                # Use Pillow to open the image file
                img = Image.open(self.image_path)
                
                # Use an in-memory buffer to save the compressed image
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=75)
                
                # Encode the compressed image bytes to a Base64 string
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Push the data to all connected web clients via the global server instance
                if socketio_server:
                    socketio_server.emit('new_frame', {'image': img_str})

            except IOError:
                # It's possible to catch the file while it's being written.
                # In that case, we just skip this frame and wait for the next event.
                # print(f"Could not read image file (still being written?): {self.image_path}")
                pass
            except Exception as e:
                print(f"Error processing image: {e}")

def start_file_watcher(path, filename):
    """Initializes and starts the file system observer in a blocking loop."""
    event_handler = ImageChangeHandler(os.path.join(path, filename))
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"[*] Started watching for changes to '{filename}' in '{path}'")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ==============================================================================
#  2. FLASK WEB SERVER LOGIC
# ==============================================================================

def create_app():
    """Creates the Flask application and the SocketIO server."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'a_secret_key_for_rl_sessions!'
    
    # Assign the created SocketIO server to the global variable
    global socketio_server
    socketio_server = SocketIO(app)

    @app.route('/')
    def index():
        """Serves the main HTML page."""
        return render_template('index_socket.html')

    @socketio_server.on('connect')
    def on_connect():
        """A handler for when a new client connects via WebSocket."""
        print('Client connected to WebSocket')

    return app, socketio_server

# ==============================================================================
#  3. MAIN EXECUTION SCRIPT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A high-performance WebSocket server that monitors an image file for changes.")
    parser.add_argument('image_file', type=str, help="Path to the image file to be monitored.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="The host to run the web server on.")
    parser.add_argument('--port', type=int, default=5000, help="The port to run the web server on.")
    args = parser.parse_args()

    # Get the absolute path for reliability
    image_file_path = os.path.abspath(args.image_file)
    image_dir = os.path.dirname(image_file_path)
    image_name = os.path.basename(image_file_path)

    if not os.path.isdir(image_dir):
        print(f"Error: The directory '{image_dir}' does not exist. Please create it or check the path.")
        exit(1)

    # Create the Flask app and the SocketIO server
    app, socketio = create_app()

    # Start the file watcher in a separate, non-blocking background thread
    watcher_thread = Thread(target=start_file_watcher, args=(image_dir, image_name), daemon=True)
    watcher_thread.start()

    # Run the Flask-SocketIO web server in the main thread
    print("--- WebSocket Image Viewer ---")
    print(f"[*] Monitoring image: {image_file_path}")
    print(f"--> Open your browser to http://localhost:{args.port}")
    print("------------------------------------")
    socketio.run(app, host=args.host, port=args.port)