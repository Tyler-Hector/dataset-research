import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_folder = r'C:\Users\debas\AI trajectory\dataset-research\DataSet1\day1\7days1\processed_data\test'
file_pattern = 'TRAJ_{}.csv'
validated_folder = os.path.join(data_folder, 'validated')
os.makedirs(validated_folder, exist_ok=True)
validated_files_log = os.path.join(validated_folder, 'validated_files.txt')

def load_csv(file_idx):
    path = os.path.join(data_folder, file_pattern.format(file_idx))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def save_validated_rows(file_idx, df_validated):
    save_path = os.path.join(validated_folder, f'validated_TRAJ_{file_idx}.csv')
    df_validated = df_validated.sort_index()
    with open(save_path, 'w') as f:
        f.write('--------- rows here to her\n')
        df_validated.to_csv(f, index=False)
    print(f"Saved {len(df_validated)} validated rows to {save_path}")

def log_validated_file(file_idx):
    with open(validated_files_log, 'a') as f:
        f.write(f"TRAJ_{file_idx}.csv\n")

def plot_trajectory(df, current_row, file_idx):
    x = df['lat']
    y = df['lon']
    z = df['alt']

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, label='Trajectory')
    ax.scatter(x.iloc[current_row], y.iloc[current_row], z.iloc[current_row], color='red', s=100, label='Current Point')
    ax.set_title(f"Trajectory from file TRAJ_{file_idx}.csv, row {current_row+1} of {len(df)}")
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude')
    ax.legend()
    plt.show()

def process_files(start_index=1):
    file_index = start_index
    validated_files = []

    while True:
        df = load_csv(file_index)
        if df is None:
            print("No more files found. Exiting.")
            break

        validated_rows = []
        n_rows = len(df)
        for row_index in range(n_rows):
            plot_trajectory(df, row_index, file_index)

            current_row_data = df.iloc[row_index]

            # Your validation condition here
            if current_row_data['alt'] > 1000:
                validated_rows.append(current_row_data)

        if validated_rows:
            validated_df = pd.DataFrame(validated_rows)
            save_validated_rows(file_index, validated_df)
            log_validated_file(file_index)
            validated_files.append(file_index)

        file_index += 1

    print(f"Processed files: {file_index - start_index}")
    print(f"Validated files: {len(validated_files)} -> {validated_files}")

if __name__ == '__main__':
    process_files()
