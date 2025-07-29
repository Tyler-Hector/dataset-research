import os
import zipfile
import json
import pandas as pd

# --- 1. Define Paths ---
raw_data_folder = "raw_data"  # Folder where SCAT dataset.zip is stored
main_zip_path = os.path.join(raw_data_folder, "SCAT dataset.zip")
temp_folder = "temp_data"  # Temporary folder for extracted inner zips
output_folder = "processed_data"  # Folder to save cleaned CSVs
os.makedirs(temp_folder, exist_ok=True)   # Create temp folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True) # Create output folder if it doesn't exist

# Check if the main ZIP exists
if not os.path.exists(main_zip_path):
    print(f"Main zip not found: {main_zip_path}")
    exit()

print(f"Found main zip: {main_zip_path}")

# --- 2. Settings ---
progress_interval = 1000  # Show progress after every 1000 files processed
save_after_zip = True     # Save partial CSV after each inner ZIP processed

# --- 3. Open the main SCAT zip ---
with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
    # Find all inner ZIP files (each week)
    inner_zips = [f for f in main_zip.namelist() if f.endswith(".zip")]
    print(f"Found {len(inner_zips)} inner zip files.")

    # This will store ALL extracted data (for final combined file)
    all_data = []

    # --- 4. Loop through each inner zip ---
    for inner_zip_name in inner_zips:
        print(f"\nProcessing {inner_zip_name}...")
        
        # Extract the inner zip to temp folder
        inner_zip_path = main_zip.extract(inner_zip_name, temp_folder)

        # Store rows for this inner zip (weekly batch)
        zip_rows = []

        # --- 5. Open the inner zip and process JSON files ---
        with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
            # Filter: Only flight JSON files (skip airspace.json, grib_meteo.json)
            json_files = [
                f for f in inner_zip.namelist()
                if f.endswith(".json") and "airspace" not in f and "grib_meteo" not in f
            ]
            print(f"  {len(json_files)} flight JSON files inside {inner_zip_name}")

            # --- 6. Loop through each flight JSON ---
            for i, jf in enumerate(json_files):
                try:
                    # Read JSON content
                    with inner_zip.open(jf) as f:
                        data = json.load(f)
                        flight_id = data.get('id')  # Unique flight ID

                        # Extract data from 'plots' section (time series of positions)
                        if 'plots' in data:
                            for p in data['plots']:
                                lat = p.get("I062/105", {}).get("lat")
                                lon = p.get("I062/105", {}).get("lon")
                                alt = p.get("I062/136", {}).get("measured_flight_level")
                                time = p.get("time_of_track")

                                # Only add row if all fields exist
                                if lat and lon and alt and time:
                                    zip_rows.append({
                                        "flight_id": flight_id,
                                        "time": time,
                                        "lat": lat,
                                        "lon": lon,
                                        "alt": alt
                                    })

                except Exception as e:
                    # If any JSON file has issues, just print error and continue
                    print(f"Error in {jf}: {e}")

                # Print progress every 1000 files
                if (i + 1) % progress_interval == 0:
                    print(f"  Processed {i + 1}/{len(json_files)} files...")

        # --- 7. Add this week's data to all_data ---
        all_data.extend(zip_rows)

        # --- 8. Save partial CSV after each inner zip ---
        if save_after_zip and zip_rows:
            # Clean the zip file name for saving
            zip_name_clean = os.path.basename(inner_zip_name).replace('.zip', '').replace(' ', '_')
            partial_file = os.path.join(output_folder, f"SCAT_cleaned_{zip_name_clean}.csv")
            pd.DataFrame(zip_rows).to_csv(partial_file, index=False)
            print(f"  Saved partial file: {partial_file} ({len(zip_rows)} rows)")

# --- 9. Save final combined CSV ---
if all_data:
    df = pd.DataFrame(all_data)
    final_file = os.path.join(output_folder, "SCAT_cleaned_full.csv")
    df.to_csv(final_file, index=False)
    print(f"\n✅ Saved ALL data: {final_file} ({len(all_data)} rows total)")
else:
    print("No valid data extracted.")