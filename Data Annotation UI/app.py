import os
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from predictor import predict_from_csv

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/augmentation')
def augmentation_page():
    return render_template('augmentation.html')

@app.route('/aboutus')
def aboutus_page():
    return render_template('aboutus.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        print(f"Running prediction on file: {filepath}")
        sample_predictions, metrics = predict_from_csv(filepath)
        print("Prediction success, plot saved as: static/prediction_plot.png")
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'sample_predictions': sample_predictions,
        'metrics': metrics,
        
    })



if __name__ == '__main__':
    app.run(debug=True)
