import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from pathlib import Path


def extract_satellite_id(filename):
    match = re.search(r'(STARLINK-\d+)', filename)
    if match:
        return match.group(1)
    return None


def parse_starlink_file(file_path):
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue

                parts = line.split()
                if len(parts) >= 8:
                    try:
                        timestamp = float(parts[0])
                        datetime_str = parts[1] + ' ' + parts[2]
                        x = float(parts[3])
                        y = float(parts[4])
                        z = float(parts[5])
                        vx = float(parts[6])
                        vy = float(parts[7])
                        vz = float(parts[8])

                        data.append({
                            'timestamp': timestamp,
                            'datetime': datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f'),
                            'x': x,
                            'y': y,
                            'z': z,
                            'vx': vx,
                            'vy': vy,
                            'vz': vz
                        })
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def find_overlapping_data(df1, df2, time_tolerance=1.0):
    if df1.empty or df2.empty:
        return pd.DataFrame(), pd.DataFrame()

    matched_data1 = []
    matched_data2 = []

    for _, row1 in df1.iterrows():
        time_diff = abs(df2['timestamp'] - row1['timestamp'])
        if time_diff.empty:
            continue

        min_diff_idx = time_diff.idxmin()
        min_diff = time_diff.loc[min_diff_idx]

        if min_diff <= time_tolerance:
            matched_data1.append(row1)
            matched_data2.append(df2.loc[min_diff_idx])

    return pd.DataFrame(matched_data1), pd.DataFrame(matched_data2)


def calculate_errors(df1, df2):
    if df1.empty or df2.empty:
        return pd.DataFrame()

    errors = pd.DataFrame()

    errors['x_error'] = df2['x'].values - df1['x'].values
    errors['y_error'] = df2['y'].values - df1['y'].values
    errors['z_error'] = df2['z'].values - df1['z'].values

    errors['vx_error'] = df2['vx'].values - df1['vx'].values
    errors['vy_error'] = df2['vy'].values - df1['vy'].values
    errors['vz_error'] = df2['vz'].values - df1['vz'].values

    errors['position_error'] = np.sqrt(errors['x_error'] ** 2 +
                                       errors['y_error'] ** 2 +
                                       errors['z_error'] ** 2)

    errors['velocity_error'] = np.sqrt(errors['vx_error'] ** 2 +
                                       errors['vy_error'] ** 2 +
                                       errors['vz_error'] ** 2)

    errors['datetime'] = df1['datetime'].values
    errors['timestamp'] = df1['timestamp'].values

    return errors


def calculate_statistics(errors):
    if errors.empty:
        return {}

    stats = {}

    error_columns = ['x_error', 'y_error', 'z_error', 'vx_error', 'vy_error', 'vz_error',
                     'position_error', 'velocity_error']

    for col in error_columns:
        if col in errors.columns:
            stats[col] = {
                'mean': errors[col].mean(),
                'std': errors[col].std(),
                'min': errors[col].min(),
                'max': errors[col].max(),
                'rms': np.sqrt(np.mean(errors[col] ** 2))
            }

    return stats


def plot_errors(errors, satellite_id, save_path=None):
    if errors.empty:
        print(f"Satellite {satellite_id} has no error data, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{satellite_id} Orbit Error Analysis', fontsize=16)

    axes[0, 0].plot(errors['datetime'], errors['x_error'], label='X Error', alpha=0.7)
    axes[0, 0].plot(errors['datetime'], errors['y_error'], label='Y Error', alpha=0.7)
    axes[0, 0].plot(errors['datetime'], errors['z_error'], label='Z Error', alpha=0.7)
    axes[0, 0].set_title('Position Error Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Error (km)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(errors['datetime'], errors['vx_error'], label='Vx Error', alpha=0.7)
    axes[0, 1].plot(errors['datetime'], errors['vy_error'], label='Vy Error', alpha=0.7)
    axes[0, 1].plot(errors['datetime'], errors['vz_error'], label='Vz Error', alpha=0.7)
    axes[0, 1].set_title('Velocity Error Time Series')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Error (km/s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(errors['datetime'], errors['position_error'], color='red', linewidth=2)
    axes[1, 0].set_title('Total Position Error')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Error (km)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(errors['datetime'], errors['velocity_error'], color='blue', linewidth=2)
    axes[1, 1].set_title('Total Velocity Error')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Error (km/s)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error plot saved to: {save_path}")

    plt.close('all')


def find_satellite_files(folder1, folder2):
    files1 = glob.glob(os.path.join(folder1, "*.txt"))
    files2 = glob.glob(os.path.join(folder2, "*.txt"))

    satellite_files1 = {}
    satellite_files2 = {}

    for file_path in files1:
        filename = os.path.basename(file_path)
        satellite_id = extract_satellite_id(filename)
        if satellite_id:
            satellite_files1[satellite_id] = file_path

    for file_path in files2:
        filename = os.path.basename(file_path)
        satellite_id = extract_satellite_id(filename)
        if satellite_id:
            satellite_files2[satellite_id] = file_path

    common_satellites = set(satellite_files1.keys()) & set(satellite_files2.keys())

    satellite_pairs = {}
    for satellite_id in common_satellites:
        satellite_pairs[satellite_id] = {
            'file1': satellite_files1[satellite_id],
            'file2': satellite_files2[satellite_id]
        }

    return satellite_pairs


def process_single_satellite(satellite_id, file1_path, file2_path, output_folder):
    print(f"\nProcessing satellite: {satellite_id}")
    print(f"  File1: {os.path.basename(file1_path)}")
    print(f"  File2: {os.path.basename(file2_path)}")

    df1 = parse_starlink_file(file1_path)
    df2 = parse_starlink_file(file2_path)

    if df1.empty or df2.empty:
        print(f"  Warning: File parsing failed, skipping satellite {satellite_id}")
        return None, None

    print(f"  File1 data points: {len(df1)}")
    print(f"  File2 data points: {len(df2)}")

    matched_df1, matched_df2 = find_overlapping_data(df1, df2)

    if len(matched_df1) == 0:
        print(f"  Warning: No overlapping time period data found, skipping satellite {satellite_id}")
        return None, None

    print(f"  Overlapping data points: {len(matched_df1)}")

    errors = calculate_errors(matched_df1, matched_df2)
    stats = calculate_statistics(errors)

    if errors.empty:
        print(f"  Warning: Error calculation failed, skipping satellite {satellite_id}")
        return None, None

    output_base = os.path.join(output_folder, f"{satellite_id}_error_analysis")
    plot_path = f"{output_base}.png"
    errors_path = f"{output_base}_errors.csv"
    stats_path = f"{output_base}_statistics.txt"

    plot_errors(errors, satellite_id, plot_path)
    plt.close('all')

    errors.to_csv(errors_path, index=False)
    print(f"  Detailed error data saved to: {errors_path}")

    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {satellite_id} Orbit Error Analysis Report ===\n\n")
        f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"File1: {file1_path}\n")
        f.write(f"File2: {file2_path}\n")
        f.write(f"Overlapping period: {matched_df1['datetime'].min()} to {matched_df1['datetime'].max()}\n")
        f.write(f"Data points: {len(errors)}\n\n")

        f.write("Position Error Statistics (km):\n")
        for axis in ['x', 'y', 'z']:
            col = f'{axis}_error'
            if col in stats:
                f.write(f"  {axis.upper()} axis - Mean: {stats[col]['mean']:.6f}, Std: {stats[col]['std']:.6f}, "
                        f"Min: {stats[col]['min']:.6f}, Max: {stats[col]['max']:.6f}, "
                        f"RMS: {stats[col]['rms']:.6f}\n")

        if 'position_error' in stats:
            f.write(f"  Total position error - Mean: {stats['position_error']['mean']:.6f}, "
                    f"Std: {stats['position_error']['std']:.6f}, "
                    f"Min: {stats['position_error']['min']:.6f}, "
                    f"Max: {stats['position_error']['max']:.6f}, "
                    f"RMS: {stats['position_error']['rms']:.6f}\n\n")

        f.write("Velocity Error Statistics (km/s):\n")
        for axis in ['vx', 'vy', 'vz']:
            col = f'{axis}_error'
            if col in stats:
                f.write(f"  {axis.upper()} axis - Mean: {stats[col]['mean']:.6f}, Std: {stats[col]['std']:.6f}, "
                        f"Min: {stats[col]['min']:.6f}, Max: {stats[col]['max']:.6f}, "
                        f"RMS: {stats[col]['rms']:.6f}\n")

        if 'velocity_error' in stats:
            f.write(f"  Total velocity error - Mean: {stats['velocity_error']['mean']:.6f}, "
                    f"Std: {stats['velocity_error']['std']:.6f}, "
                    f"Min: {stats['velocity_error']['min']:.6f}, "
                    f"Max: {stats['velocity_error']['max']:.6f}, "
                    f"RMS: {stats['velocity_error']['rms']:.6f}\n")

    print(f"  Statistics report saved to: {stats_path}")

    return errors, stats


def batch_process_satellites(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    print("Finding satellite file pairs...")
    satellite_pairs = find_satellite_files(folder1, folder2)

    if not satellite_pairs:
        print("No matching satellite file pairs found!")
        return {}

    print(f"Found {len(satellite_pairs)} satellite file pairs:")
    for satellite_id in satellite_pairs:
        print(f"  - {satellite_id}")

    results = {}
    success_count = 0
    total_count = len(satellite_pairs)

    for satellite_id, file_paths in satellite_pairs.items():
        try:
            errors, stats = process_single_satellite(
                satellite_id,
                file_paths['file1'],
                file_paths['file2'],
                output_folder
            )

            if errors is not None and stats is not None:
                results[satellite_id] = {
                    'errors': errors,
                    'statistics': stats,
                    'status': 'success'
                }
                success_count += 1
            else:
                results[satellite_id] = {
                    'errors': None,
                    'statistics': None,
                    'status': 'failed'
                }
        except Exception as e:
            print(f"  Error processing satellite {satellite_id}: {e}")
            results[satellite_id] = {
                'errors': None,
                'statistics': None,
                'status': 'error',
                'error_message': str(e)
            }

    summary_path = os.path.join(output_folder, "batch_processing_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Batch Satellite Orbit Error Analysis Summary Report ===\n\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference data folder: {folder1}\n")
        f.write(f"Comparison data folder: {folder2}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"Total satellites: {total_count}\n")
        f.write(f"Successfully processed: {success_count}\n")
        f.write(f"Processing failed: {total_count - success_count}\n\n")

        f.write("Processing results details:\n")
        for satellite_id, result in results.items():
            f.write(f"  {satellite_id}: {result['status']}\n")
            if result['status'] == 'error':
                f.write(f"    Error message: {result.get('error_message', 'Unknown error')}\n")

        f.write("\nSuccessfully processed satellite statistics overview:\n")
        for satellite_id, result in results.items():
            if result['status'] == 'success' and result['statistics']:
                stats = result['statistics']
                if 'position_error' in stats and 'velocity_error' in stats:
                    f.write(f"  {satellite_id}:\n")
                    f.write(f"    Position error RMS: {stats['position_error']['rms']:.6f} km\n")
                    f.write(f"    Velocity error RMS: {stats['velocity_error']['rms']:.6f} km/s\n")

    print(f"\n=== Batch processing completed ===")
    print(f"Total satellites: {total_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Processing failed: {total_count - success_count}")
    print(f"Summary report saved to: {summary_path}")

    return results


def main():
    folder1 = r"E:/gnssda100/1"
    folder2 = r"E:/gnssda100/2"
    output_folder = r"E:/gnssda100/batch_error_analysis"

    if not os.path.exists(folder1):
        print(f"Error: Folder {folder1} does not exist!")
        return

    if not os.path.exists(folder2):
        print(f"Error: Folder {folder2} does not exist!")
        return

    print("Starting batch satellite orbit error analysis...")
    print(f"Reference data folder: {folder1}")
    print(f"Comparison data folder: {folder2}")

    results = batch_process_satellites(folder1, folder2, output_folder)

    return results


if __name__ == "__main__":
    folder1 = r"E:/1"
    folder2 = r"E:/2"
    output_folder = r"E:/out_error"

    results = batch_process_satellites(folder1, folder2, output_folder)
