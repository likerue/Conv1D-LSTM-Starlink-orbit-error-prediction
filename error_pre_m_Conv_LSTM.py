import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import gc
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class RealtimeSatelliteErrorPredictor:
    def __init__(self, sequence_length=60, conv_filters=64, conv_kernel_size=3, lstm_units=128):
        self.sequence_length = sequence_length
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.lstm_units = lstm_units
        self.scalers = {}
        self.models = {}

    def parse_line_intelligent(self, line, line_idx):
        if line.startswith('#') or line.strip() == '':
            return None

        parts = line.split()
        if len(parts) < 8:
            return None

        try:
            timestamp_str = parts[0]
            datetime_str = parts[1] + " " + parts[2]

            e_x = float(parts[3])
            e_y = float(parts[4])
            e_z = float(parts[5])
            e_vx = float(parts[6])
            e_vy = float(parts[7])
            e_vz = float(parts[8])

            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

            return [dt, e_x, e_y, e_z, e_vx, e_vy, e_vz]

        except (ValueError, IndexError) as e:
            print(f"Error parsing line {line_idx}: {line.strip()}")
            return None

    def load_single_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            filename = os.path.basename(file_path)
            print(f"Processing file: {filename}")

            data_lines = [line.strip() for line in lines
                          if line.strip() and not line.strip().startswith('#')]

            print(f"Data lines after filtering: {len(data_lines)}")

            if not data_lines:
                print(f"No valid data found in {filename}")
                return None

            data = []
            for line_idx, line in enumerate(data_lines):
                parsed_data = self.parse_line_intelligent(line, line_idx)
                if parsed_data:
                    data.append(parsed_data)

            if not data:
                print(f"No parseable data found in {filename}")
                return None

            df = pd.DataFrame(data, columns=['datetime', 'e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz'])
            df = df.sort_values('datetime').reset_index(drop=True)

            df['minute_index'] = range(len(df))

            print(f"Successfully loaded {len(df)} records from {filename}")
            print(f"Data shape: {df.shape}")
            print(f"Time range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            print("First 3 rows:")
            print(df.head(3))
            print("-" * 50)

            return df, filename

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def prepare_sequences_for_target(self, data, target_column, target_name):
        print(f"Preparing sequences for target '{target_name}', dimension '{target_column}'")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data for {target_name}")

        values = data[target_column].values.reshape(-1, 1)
        print(f"Target values shape: {values.shape}")
        print(f"Target values range: [{values.min():.3f}, {values.max():.3f}]")

        scaler_key = f"{target_name}_{target_column}"
        self.scalers[scaler_key] = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = self.scalers[scaler_key].fit_transform(values)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_values)):
            X.append(scaled_values[i - self.sequence_length:i, 0])
            y.append(scaled_values[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        print(f"Sequence shapes: X={X.shape}, y={y.shape}")
        return X, y, scaler_key

    def build_model(self):
        inputs = Input(shape=(self.sequence_length, 1))

        conv1 = Conv1D(filters=self.conv_filters,
                       kernel_size=self.conv_kernel_size,
                       activation='relu',
                       padding='same')(inputs)
        conv1 = Dropout(0.2)(conv1)

        conv2 = Conv1D(filters=self.conv_filters // 2,
                       kernel_size=self.conv_kernel_size,
                       activation='relu',
                       padding='same')(conv1)
        conv2 = Dropout(0.2)(conv2)

        pool = MaxPooling1D(pool_size=2)(conv2)

        lstm1 = LSTM(self.lstm_units,
                     return_sequences=True,
                     dropout=0.2,
                     recurrent_dropout=0.2)(pool)

        lstm2 = LSTM(self.lstm_units // 2,
                     return_sequences=False,
                     dropout=0.2,
                     recurrent_dropout=0.2)(lstm1)

        dense1 = Dense(64, activation='relu')(lstm2)
        dense1 = Dropout(0.3)(dense1)

        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)

        outputs = Dense(1, activation='tanh')(dense2)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae'])

        return model

    def train_target_model(self, X, y, target_name, dimension, train_ratio=0.8, epochs=100, batch_size=32):
        print(f"\nTraining Conv-LSTM model for target: {target_name}, dimension: {dimension}")

        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Train set: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")

        model = self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        model_key = f"{target_name}_{dimension}"
        self.models[model_key] = model

        self.evaluate_target_model(model, X_test, y_test, target_name, dimension)

        return history, X_test, y_test

    def evaluate_target_model(self, model, X_test, y_test, target_name, dimension):
        predictions = model.predict(X_test, verbose=0)

        scaler_key = f"{target_name}_{dimension}"
        scaler = self.scalers[scaler_key]

        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_orig = scaler.inverse_transform(predictions)

        mae = mean_absolute_error(y_test_orig, predictions_orig)
        mse = mean_squared_error(y_test_orig, predictions_orig)
        rmse = np.sqrt(mse)

        print(f"Conv-LSTM model performance for {target_name} - {dimension}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        return mae, mse, rmse

    def predict_future_for_target(self, target_name, dimension, last_sequence, n_minutes):
        model_key = f"{target_name}_{dimension}"
        scaler_key = f"{target_name}_{dimension}"

        if model_key not in self.models:
            raise ValueError(f"No trained model found for {target_name} - {dimension}")

        model = self.models[model_key]
        scaler = self.scalers[scaler_key]

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(n_minutes):
            next_pred = model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
            predictions.append(next_pred[0, 0])

            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        return predictions.flatten()

    def get_prediction_start_time(self, data):
        last_datetime = data['datetime'].iloc[-1]
        start_time = last_datetime + timedelta(minutes=1)
        return start_time

    def process_single_target(self, file_path, dimensions, n_hours=8,
                              train_ratio=0.8, epochs=50, batch_size=32, output_folder=None):
        print(f"\n{'=' * 60}")
        print(f"Processing single target file: {os.path.basename(file_path)}")
        print(f"{'=' * 60}")

        result = self.load_single_file(file_path)
        if result is None:
            print(f"Failed to load file: {file_path}")
            return None

        data, filename = result
        target_name = os.path.splitext(filename)[0]

        start_time = self.get_prediction_start_time(data)
        print(f"Prediction start time: {start_time}")

        n_minutes = n_hours * 60
        target_predictions = {}

        for dimension in dimensions:
            try:
                print(f"\nProcessing dimension: {dimension}")

                X, y, scaler_key = self.prepare_sequences_for_target(data, dimension, target_name)

                if len(X) < 20:
                    print(f"Warning: Not enough data for {target_name} - {dimension}, skipping...")
                    continue

                history, X_test, y_test = self.train_target_model(
                    X, y, target_name, dimension, train_ratio, epochs, batch_size
                )

                last_sequence = X_test[-1].flatten()
                future_predictions = self.predict_future_for_target(
                    target_name, dimension, last_sequence, n_minutes
                )

                target_predictions[dimension] = future_predictions
                print(f"Successfully predicted {n_minutes} minutes for {target_name} - {dimension}")

            except Exception as e:
                print(f"Error processing {target_name} - {dimension}: {e}")
                continue

        if target_predictions and output_folder:
            self.save_single_target_results(
                target_name, target_predictions, start_time, output_folder, n_hours
            )

            self.plot_single_target_results(
                target_name, target_predictions, start_time, output_folder, n_hours
            )

        return target_predictions

    def save_single_target_results(self, target_name, predictions, start_time, output_folder, n_hours):
        n_minutes = n_hours * 60

        timestamps = [start_time + timedelta(minutes=i) for i in range(n_minutes)]

        result_data = {'timestamp': timestamps}

        for dimension, pred_values in predictions.items():
            if len(pred_values) == n_minutes:
                result_data[f'{dimension}_predicted'] = pred_values
            else:
                print(f"Warning: Prediction length mismatch for {target_name} - {dimension}")
                padded_values = list(pred_values) + [np.nan] * (n_minutes - len(pred_values))
                result_data[f'{dimension}_predicted'] = padded_values[:n_minutes]

        results_df = pd.DataFrame(result_data)

        output_file = os.path.join(output_folder, f"{target_name}_error_predictions_{n_hours}h.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved predictions for {target_name}: {output_file}")

    def plot_single_target_results(self, target_name, predictions, start_time, output_folder, n_hours):
        n_minutes = n_hours * 60
        minutes = range(1, n_minutes + 1)
        hours = [m / 60.0 for m in minutes]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        dimensions = ['e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz']

        for i, dimension in enumerate(dimensions):
            if dimension in predictions:
                axes[i].plot(hours, predictions[dimension], linewidth=2, alpha=0.8, color='red')
                axes[i].set_title(f'{target_name} - {dimension.upper()} Error Prediction (Conv-LSTM)', fontsize=12)
                axes[i].set_xlabel('Hours', fontsize=10)
                axes[i].set_ylabel(f'{dimension.upper()} Error Value', fontsize=10)
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'No data for {dimension.upper()}',
                             transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{target_name} - {dimension.upper()} Error Prediction (Conv-LSTM)', fontsize=12)

        plt.tight_layout()

        plot_file = os.path.join(output_folder, f'{target_name}_conv_lstm_error_predictions_{n_hours}h.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for {target_name}: {plot_file}")

    def batch_process_all_targets(self, input_folder, output_folder, dimensions, n_hours=8,
                                  train_ratio=0.8, epochs=50, batch_size=32):
        os.makedirs(output_folder, exist_ok=True)

        txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} txt files in {input_folder}")

        processed_count = 0
        failed_count = 0

        for i, filename in enumerate(txt_files, 1):
            file_path = os.path.join(input_folder, filename)

            print(f"\n{'=' * 80}")
            print(f"Processing file {i}/{len(txt_files)}: {filename}")
            print(f"{'=' * 80}")

            try:
                result = self.process_single_target(
                    file_path, dimensions, n_hours, train_ratio, epochs, batch_size, output_folder
                )

                if result:
                    processed_count += 1
                    print(f"Successfully processed and saved results for: {filename}")
                else:
                    failed_count += 1
                    print(f"Failed to process: {filename}")

                tf.keras.backend.clear_session()
                plt.close('all')
                gc.collect()

            except Exception as e:
                failed_count += 1
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 80}")
        print("BATCH PROCESSING SUMMARY (Conv-LSTM)")
        print(f"{'=' * 80}")
        print(f"Total files: {len(txt_files)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"Results saved to: {output_folder}")

        return processed_count, failed_count


def main():
    tmp_dir = "path/to/temp"
    os.makedirs(tmp_dir, exist_ok=True)

    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    tempfile.tempdir = tmp_dir

    os.environ["TFHUB_CACHE_DIR"] = os.path.join(tmp_dir, "tfhub")
    os.environ["MPLCONFIGDIR"] = os.path.join(tmp_dir, "matplotlib")

    input_folder = "path/to/input/out_error_8h/"
    output_folder = "path/to/output/out_error_8h_pre_conv_lstm/"
    dimensions = ["e_x", "e_y", "e_z", "e_vx", "e_vy", "e_vz"]
    n_hours = 8

    sequence_length = 120
    conv_filters = 64
    conv_kernel_size = 3
    lstm_units = 128
    epochs = 100
    batch_size = 32
    train_ratio = 0.8

    os.makedirs(output_folder, exist_ok=True)

    predictor = RealtimeSatelliteErrorPredictor(
        sequence_length=sequence_length,
        conv_filters=conv_filters,
        conv_kernel_size=conv_kernel_size,
        lstm_units=lstm_units
    )

    try:
        print("Starting realtime batch satellite error prediction with Conv-LSTM...")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Prediction hours: {n_hours}")
        print(f"Dimensions to predict: {dimensions}")
        print(f"Model: Conv-LSTM (filters={conv_filters}, kernel_size={conv_kernel_size}, lstm_units={lstm_units})")

        processed_count, failed_count = predictor.batch_process_all_targets(
            input_folder, output_folder, dimensions, n_hours, train_ratio, epochs, batch_size
        )

        print(f"\n{'=' * 80}")
        print("REALTIME BATCH CONV-LSTM ERROR PREDICTION COMPLETED!")
        print(f"{'=' * 80}")
        print(f"Successfully processed: {processed_count} files")
        print(f"Failed: {failed_count} files")
        print(f"All results saved to: {output_folder}")

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()
