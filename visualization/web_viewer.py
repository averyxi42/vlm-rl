import argparse
import os
from flask import Flask, render_template, send_from_directory

# The directory where the web_viewer.py script itself is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app(image_directory, image_filename):
    """Creates the Flask application."""
    app = Flask(__name__, template_folder=os.path.join(SCRIPT_DIR, 'templates'))

    @app.route('/')
    def index():
        """Serves the main HTML page."""
        return render_template('index.html', image_name=image_filename)

    @app.route(f'/{image_filename}')
    def get_image():
        """Serves the latest training image."""
        try:
            return send_from_directory(image_directory, image_filename)
        except FileNotFoundError:
            return "Image not found yet. Waiting for training to start...", 404

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A robust real-time web viewer for a frequently updated image file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'image_file',
        type=str,
        help="Path to the image file to monitor.\nCan be relative (e.g., 'outputs/frame.png') or absolute."
    )
    parser.add_argument('--host', type=str, default='0.0.0.0', help="The host to run the web server on.")
    parser.add_argument('--port', type=int, default=5000, help="The port to run the web server on.")
    args = parser.parse_args()

    image_file_path = os.path.abspath(args.image_file)
    image_dir = os.path.dirname(image_file_path)
    image_name = os.path.basename(image_file_path)

    if not os.path.isdir(image_dir):
        print(f"Warning: The directory '{image_dir}' does not exist. Your training script will need to create it.")

    app = create_app(image_dir, image_name)

    print("--- Real-time Training Viewer ---")
    print(f"[*] Monitoring image: {image_file_path}")
    print(f"--> Open your browser to: http://localhost:{args.port}")
    print("------------------------------------")
    
    # --- THIS IS THE FIX ---
    # The problematic `extra_files` argument has been removed.
    app.run(host=args.host, port=args.port)