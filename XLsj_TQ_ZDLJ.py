#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime
import glob


def parse_timestamp(timestamp_str):
    try:
        year = int(timestamp_str[:4])
        day_of_year = int(timestamp_str[4:7])
        hour = int(timestamp_str[7:9])
        minute = int(timestamp_str[9:11])
        second = float(timestamp_str[11:])

        base_date = datetime(year, 1, 1)
        target_date = base_date.replace(day=1).replace(month=1)
        from datetime import timedelta
        target_date = target_date + timedelta(days=day_of_year - 1)
        target_date = target_date.replace(hour=hour, minute=minute)

        return target_date.strftime("%Y-%m-%d %H:%M:%S") + f".{int((second % 1) * 1000):03d}"
    except:
        return timestamp_str


def extract_ephemeris_data(input_file):
    data_points = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == 'UVW':
                data_start = i + 1
                break

        i = data_start
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            parts = line.split()
            if len(parts) >= 7:
                timestamp = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vx, vy, vz = float(parts[4]), float(parts[5]), float(parts[6])

                readable_time = parse_timestamp(timestamp)

                data_points.append({
                    'timestamp_raw': timestamp,
                    'timestamp': readable_time,
                    'position': [x, y, z],
                    'velocity': [vx, vy, vz]
                })

            i += 4

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return []

    return data_points


def write_output_file(data_points, output_file, input_filename):
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Starlink Satellite Ephemeris Data Extraction Result\n")
            f.write(f"# Source file: {input_filename}\n")
            f.write(f"# Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Number of data points: {len(data_points)}\n")
            f.write("#\n")
            f.write("# Format description:\n")
            f.write(
                "# Timestamp(raw)    DateTime                X(km)      Y(km)      Z(km)      Vx(km/s)   Vy(km/s)   Vz(km/s)\n")
            f.write("#" + "=" * 120 + "\n")

            for point in data_points:
                f.write(f"{point['timestamp_raw']:<18} {point['timestamp']:<23} ")
                f.write(f"{point['position'][0]:>10.6f} {point['position'][1]:>10.6f} {point['position'][2]:>10.6f} ")
                f.write(f"{point['velocity'][0]:>10.6f} {point['velocity'][1]:>10.6f} {point['velocity'][2]:>10.6f}\n")

            f.write("\n# Statistics:\n")
            if data_points:
                pos_data = [p['position'] for p in data_points]
                vel_data = [p['velocity'] for p in data_points]

                x_coords = [p[0] for p in pos_data]
                y_coords = [p[1] for p in pos_data]
                z_coords = [p[2] for p in pos_data]

                f.write(f"# X coordinate range: {min(x_coords):.3f} ~ {max(x_coords):.3f} km\n")
                f.write(f"# Y coordinate range: {min(y_coords):.3f} ~ {max(y_coords):.3f} km\n")
                f.write(f"# Z coordinate range: {min(z_coords):.3f} ~ {max(z_coords):.3f} km\n")

                altitudes = [(x ** 2 + y ** 2 + z ** 2) ** 0.5 - 6371 for x, y, z in pos_data]
                f.write(f"# Orbital altitude estimate: {min(altitudes):.1f} ~ {max(altitudes):.1f} km\n")

                speeds = [(vx ** 2 + vy ** 2 + vz ** 2) ** 0.5 for vx, vy, vz in vel_data]
                f.write(f"# Orbital speed: {min(speeds):.3f} ~ {max(speeds):.3f} km/s\n")

        print(f"Successfully processed: {output_file} ({len(data_points)} data points)")
        return True

    except Exception as e:
        print(f"Error writing file {output_file}: {e}")
        return False


def process_directory(input_dir, output_dir=None):
    pattern = os.path.join(input_dir, "*.txt")
    txt_files = glob.glob(pattern)

    if not txt_files:
        print(f"No txt files found in directory {input_dir}")
        return

    if output_dir is None:
        output_dir = input_dir
        print(f"Using input directory as output directory: {output_dir}")
    else:
        print(f"Output directory: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    print(f"Found {len(txt_files)} txt files")
    print("Starting batch processing...\n")

    success_count = 0
    total_points = 0

    for input_file in txt_files:
        print(f"Processing file: {os.path.basename(input_file)}")

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_out.txt")

        data_points = extract_ephemeris_data(input_file)

        if data_points:
            if write_output_file(data_points, output_file, os.path.basename(input_file)):
                success_count += 1
                total_points += len(data_points)
        else:
            print(f"No valid data found in file {input_file}")

        print()

    print("=" * 60)
    print(f"Batch processing completed!")
    print(f"Total files: {len(txt_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed files: {len(txt_files) - success_count}")
    print(f"Total data points: {total_points}")
    print(f"Output location: {output_dir}")


def get_valid_directory(prompt, check_exists=True):
    while True:
        dir_path = input(prompt).strip()

        if not dir_path:
            return None

        dir_path = os.path.abspath(dir_path)

        if check_exists:
            if not os.path.exists(dir_path):
                print(f"Error: Directory {dir_path} does not exist!")
                continue
            if not os.path.isdir(dir_path):
                print(f"Error: {dir_path} is not a directory!")
                continue
        else:
            parent_dir = os.path.dirname(dir_path)
            if parent_dir and not os.path.exists(parent_dir):
                print(f"Error: Parent directory {parent_dir} does not exist!")
                continue

        return dir_path


def main():
    print("Starlink Ephemeris Data Extractor (with custom output path support)")
    print("=" * 60)

    print("Step 1: Specify input directory")
    input_dir = get_valid_directory("Please enter the directory path containing ephemeris txt files: ", check_exists=True)

    if input_dir is None:
        print("Operation cancelled")
        return

    print(f"Input directory: {input_dir}")
    print()

    print("Step 2: Specify output directory")
    print("Tip: Press Enter directly to use the input directory as the output directory")
    output_dir = get_valid_directory("Please enter output directory path (optional): ", check_exists=False)

    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print("Will use input directory as output directory")

    print()

    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()
