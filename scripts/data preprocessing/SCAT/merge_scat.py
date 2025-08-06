import os
import zipfile
import json
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

# define paths
raw_data_folder = "raw_data"  # folder where SCAT dataset.zip is
main_zip_path = os.path.join(raw_data_folder, "SCAT dataset.zip")
temp_folder = "temp_data"  # temp folder for extracted inner zips
output_folder = "processed_data"  # folder to save cleaned CSVs

print(f"Found main zip: {main_zip_path}")

# settings
progress_interval = 1000  # show progress after every 1000 files processed
save_after_zip = False     # save partial CSV after each inner ZIP processed

# open main SCAT zip 
with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
    inner_zips = [f for f in main_zip.namelist() if f.endswith(".zip")]
    print(f"Found {len(inner_zips)} inner zip files.")

    all_data = []

    # loop through each inner zip
    for inner_zip_name in inner_zips:
        print(f"\nProcessing {inner_zip_name}...")
        inner_zip_path = main_zip.extract(inner_zip_name, temp_folder)

        zip_rows = []

        with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
            # load weather file for the week
            weather_file = [f for f in inner_zip.namelist() if "grib_meteo" in f]
            if not weather_file:
                print("  No weather file found. Skipping weather merge for this week.")
                continue

            with inner_zip.open(weather_file[0]) as wf:
                weather_data = json.load(wf)
            weather_df = pd.DataFrame(weather_data)
            weather_df["time"] = pd.to_datetime(weather_df["time"], format="ISO8601", errors="coerce")
            weather_df["alt_ft"] = weather_df["alt"] * 100

            # KDTree for spatial match
            weather_coords = np.array(list(zip(weather_df["lat"], weather_df["lon"])))
            tree = cKDTree(weather_coords)

            # only flight JSON files (skip airspace.json & weather)
            json_files = [
                f for f in inner_zip.namelist()
                if f.endswith(".json") and "airspace" not in f and "grib_meteo" not in f
            ]
            print(f"  {len(json_files)} flight JSON files inside {inner_zip_name}")

            # process each flight JSON 
            for i, jf in enumerate(json_files):
                try:
                    with inner_zip.open(jf) as f:
                        data = json.load(f)
                        flight_id = data.get('id')

                        if 'plots' in data:
                            flight_rows = []
                            for p in data['plots']:
                                lat = p.get("I062/105", {}).get("lat")
                                lon = p.get("I062/105", {}).get("lon")
                                alt = p.get("I062/136", {}).get("measured_flight_level")
                                time = p.get("time_of_track")

                                if lat and lon and alt and time:
                                    flight_rows.append({
                                        "flight_id": flight_id,
                                        "time": time,
                                        "lat": lat,
                                        "lon": lon,
                                        "alt": alt
                                    })

                            if flight_rows:
                                flight_df = pd.DataFrame(flight_rows)
                                flight_df["time"] = pd.to_datetime(flight_df["time"], format="ISO8601", errors="coerce")

                                # Merge weather (nearest lat/lon)
                                traj_coords = np.array(list(zip(flight_df["lat"], flight_df["lon"])))
                                _, idx = tree.query(traj_coords, k=1)

                                flight_df["temp"] = weather_df.iloc[idx]["temp"].values
                                flight_df["wind_spd"] = weather_df.iloc[idx]["wind_spd"].values
                                flight_df["wind_dir"] = weather_df.iloc[idx]["wind_dir"].values

                                zip_rows.extend(flight_df.to_dict("records"))

                except Exception as e:
                    print(f"Error in {jf}: {e}")

                if (i + 1) % progress_interval == 0:
                    print(f"  Processed {i + 1}/{len(json_files)} files...")

        all_data.extend(zip_rows)

        # save partial CSV for this week
        if save_after_zip and zip_rows:
            zip_name_clean = os.path.basename(inner_zip_name).replace('.zip', '').replace(' ', '_')
            partial_file = os.path.join(output_folder, f"SCAT_cleaned_{zip_name_clean}.csv")
            pd.DataFrame(zip_rows).to_csv(partial_file, index=False)
            print(f"  Saved partial file: {partial_file} ({len(zip_rows)} rows)")

# save final combined CSV
if all_data:
    df = pd.DataFrame(all_data)
    final_file = os.path.join(output_folder, "SCAT_cleaned_full.csv")
    df.to_csv(final_file, index=False)
    print(f"\nSaved ALL data: {final_file} ({len(all_data)} rows total)")
else:
    print("No valid data extracted.")