from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')
import io
import os  # <-- For creating directories and file handling

app = Flask(__name__)

# Load data only once
data = pd.read_csv('Data Preprocessing/final/combo_Synthetic-Oslo_1.csv').to_numpy()
index = [0]

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/augmentation')
def augmentation_page():
    return render_template('augmentation.html')
@app.route('/aboutus')
def aboutus_page():
    return render_template('aboutus.html')

@app.route('/plot')
def plot():
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
    content = request.get_json()
    valid = content.get("valid")  # 'yes' or 'no'
    print(request)
    i = index[0]
    row = data[i]
    x, y, z = row[2], row[1], row[3]

    if valid == "yes":
        save_path = 'validated_data/'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # creates folder if needed
        pd.DataFrame(data, columns=['lat','lon','alt','time']).to_csv(save_path+'combo_Synthetic-Oslo_1.csv', index=False)
        print(f"Saved coords: {x}, {y}, {z} to {save_path}")

    index[0] += 1
    if index[0] >= len(data):
        index[0] = 0

    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
