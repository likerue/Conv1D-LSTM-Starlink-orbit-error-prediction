import os
import pandas as pd
from datetime import datetime, timedelta
import glob


def extract_first_8_hours_data(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            base_filename = os.path.splitext(os.path.basename(csv_file))[0]

            df['datetime'] = pd.to_datetime(df['datetime'])

            df = df.sort_values('datetime')

            start_time = df['datetime'].iloc[0]

            end_time = start_time + timedelta(hours=8)

            filtered_df = df[df['datetime'] <= end_time]

            output_filename = f"{base_filename}_8h_out.txt"
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Starlink Satellite Ephemeris Data Extraction Result\n")
                f.write(f"# Source file: {os.path.basename(csv_file)}\n")
                f.write(f"# Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Number of data points: {len(filtered_df)}\n")
                f.write("#\n")
                f.write("# Format description:\n")
                f.write(
                    "# Timestamp(raw)    DateTime                e_X(m)      e_Y(m)      e_Z(m)      e_Vx(m/s)   e_Vy(m/s)   e_Vz(m/s)\n")
                f.write("#" + "=" * 120 + "\n")

                for _, row in filtered_df.iterrows():
                    timestamp = row['timestamp']
                    datetime_str = row['datetime'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    x = row['x_error'] * 1000
                    y = row['y_error'] * 1000
                    z = row['z_error'] * 1000
                    vx = row['vx_error'] * 1000
                    vy = row['vy_error'] * 1000
                    vz = row['vz_error'] * 1000

                    f.write(
                        f"{timestamp:>18.3f}  {datetime_str} {x:>12.6f} {y:>12.6f} {z:>12.6f}  {vx:>10.6f}  {vy:>10.6f}  {vz:>10.6f}\n")

            print(f"Processed: {os.path.basename(csv_file)} -> {output_filename}")
            print(f"  Original data points: {len(df)}, Extracted data points: {len(filtered_df)}")
            print(f"  Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
            continue

    print(f"\nProcessing completed! All output files saved to: {output_folder}")


def main():
    input_folder = r"E:/out_error/"
    output_folder = r"E:/out_error_8h/"

    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        print("Please modify the input_folder variable to the correct path")
        return

    extract_first_8_hours_data(input_folder, output_folder)


if __name__ == "__main__":
    main()
