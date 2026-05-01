from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import sys
import subprocess

app = Flask(__name__)
CORS(app)

# Get the directory where this script lives
APP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(APP_DIR, 'run_tryon.py')


@app.route('/api/infer', methods=['POST'])
def infer():
    # Check that the post contains the right fields
    if not all(i in request.files for i in ['person', 'clothing']):
        return jsonify(error='Incorrect payload! Need "person" and "clothing" files.')

    # Check that the files are images
    person_mime = request.files['person'].mimetype or ''
    clothing_mime = request.files['clothing'].mimetype or ''
    if not (person_mime.startswith('image') and clothing_mime.startswith('image')):
        return jsonify(error='Inputs must be images!')

    # Save input files
    inputs_dir = os.path.join(APP_DIR, 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)

    person_path = os.path.join(inputs_dir, 'input_person.jpg')
    clothing_path = os.path.join(inputs_dir, 'input_clothing.jpg')

    request.files['person'].save(person_path)
    request.files['clothing'].save(clothing_path)

    # Run the try-on pipeline
    try:
        result = subprocess.run(
            [sys.executable, RUNNER, person_path, clothing_path],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=APP_DIR,
        )
        output_log = result.stdout + '\n' + result.stderr
        print(f"[Flask] run_tryon stdout:\n{result.stdout}")
        if result.returncode != 0:
            print(f"[Flask] run_tryon stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        return jsonify(error='Timeout: Try-on pipeline took too long.')
    except Exception as e:
        return jsonify(error=f'Failed to run pipeline: {str(e)}')

    # Check that output exists
    output_path = os.path.join(APP_DIR, 'output', 'output.png')
    if not os.path.isfile(output_path):
        return jsonify(error='500: Internal server error (pipeline produced no output)')

    return send_from_directory(os.path.join(APP_DIR, 'output'), 'output.png')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify(status='ok')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
