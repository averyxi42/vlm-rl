from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    """A simple route that returns a success message."""
    print("--- Request received at / route ---")
    return "<h1>Success! The test server is running.</h1>"

if __name__ == '__main__':
    print("\nStarting minimal Flask server...")
    print(">>> It should be available at http://localhost:5000 <<<")
    
    # Run the app directly on all interfaces on port 5000
    # This will block the terminal until you stop it with Ctrl+C
    app.run(host='0.0.0.0', port=8000)