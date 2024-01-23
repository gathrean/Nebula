import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

@app.route('/execute-python', methods=['POST'])
def execute_python():
    data = request.get_json()
    audio_file_path = data.get('audioFilePath')

    # Log request details
    logging.info(f"Received request with audioFilePath: {audio_file_path}")
    logging.info(f"Request headers: {request.headers}")

    # Path to your Python script
    python_script_path = os.path.join(os.path.dirname(__file__), 'inference.py')

    try:
        # Execute the Python script using subprocess
        result = subprocess.run(['python3', python_script_path, audio_file_path], check=True, capture_output=True, text=True)
        
        # Assuming the output contains predicted labels
        predicted_labels = result.stdout.strip().split('\n')

        # Return the predicted labels in the response
        return jsonify(success=True, predicted=predicted_labels)
    except Exception as e:
        print(f'Error executing Python script: {e}')
        return jsonify(success=False, error='Internal Server Error')

if __name__ == '__main__':
    app.run(port=8000)
