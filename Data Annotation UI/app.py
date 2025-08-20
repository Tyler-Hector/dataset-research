import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from predictor import predict_from_csv  # Updated predictor
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import io
import os  # <-- For creating directories and file handling


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


# IMPORTANT VARIABLES ------------------------------------------------
dataset="Oslo"
start=1


count=start
dataset_num=-1
if(dataset=="AirLab"):
    dataset_num=0
elif(dataset=="Oslo"):
    dataset_num=1
elif(dataset=="Synthetic"):
    dataset_num=2
else:
    dataset_num=3

# Load data only once
data = pd.read_csv('Data Preprocessing/'+dataset+'/processed_data/TRAJ_'+str(count)+'.csv').to_numpy()
max = [3088,6272,5731,5000]
index=[0]

# ----- Existing pages -----
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

# -------------------------------
# Annotation UI routes
# -------------------------------

@app.route('/plot')
def plot():
    global data
    count = request.args.get("count", type=int)
    print("Plotting trajectory:", count)  # <-- This should appear in your terminal
    
    i = index[0]

    if data.shape[0] < 2:
        return jsonify(error="Not enough data to plot a trajectory.")

    x = data[:, 0]  # X column
    y = data[:, 1]  # Y column
    z = data[:, 2]  # Z column

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot a smooth line (remove the marker)
    ax.plot(x, y, z, color='blue', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Trajectory from row {i} to {i+len(x)-1}")
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/next', methods=['POST'])
def next_point():
    global data, count
    content = request.get_json()
    valid = content.get("valid")  # 'yes' or 'no'
    print(request)
    i = index[0]
    row = data[i]
    x, y, z = row[2], row[1], row[3]
    
    if valid == "yes":
        save_path = 'validated_data/'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # creates folder if needed
        pd.DataFrame(data, columns=['lat','lon','alt','time']).to_csv(save_path+dataset+str(count)+'.csv', index=False)
        print(f"Saved coords: {x}, {y}, {z} to {save_path}")

    index[0] += 1
    if index[0] >= len(data):
        index[0] = 0
    
    count+=1
    if(count>=max[dataset_num]):
        sys.exit()
    data = pd.read_csv('Data Preprocessing/'+dataset+'/processed_data/TRAJ_'+str(count)+'.csv').to_numpy()
    
    return jsonify(success=True)


# ----- Updated upload_predict -----
@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    # Save file with unique name
    filename = str(uuid.uuid4()) + ".csv"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        sample_predictions, metrics, plot_path = predict_from_csv(filepath)
        app.logger.info(f"Prediction success, plot saved as: {plot_path}")
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'sample_predictions': sample_predictions,
        'metrics': metrics,
        'plot_url': url_for('static', filename=f'plots/{os.path.basename(plot_path)}')
    })

if __name__ == "__main__":
    app.run(debug=True)
