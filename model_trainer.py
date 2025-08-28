# model_trainer.py
# The Back-End Brain: A production-grade analytical engine.
# It runs in a separate process, continuously ingesting data from the
# database, training state-of-the-art models, and providing them to
# the main trading bot for real-time predictions.

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import time
import os
import warnings
import joblib
import sys
import re
import sqlite3
import pickle
import subprocess
from collections import deque, Counter, defaultdict
from scipy import stats
from typing import Dict, Deque, Tuple, Any, Optional
from multiprocessing import Queue
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from scipy.stats import linregress
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import xgboost as xgb
from catboost import CatBoostClassifier
# CORRECTION START
from tensorflow.keras.models import load_model
# CORRECTION END

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# --- Self-Sufficiency and Dependency Management ---
# This ensures a flawless setup in any environment by installing dependencies
# before anything else. This is our commitment to quality.
def install_and_import(package, install_name=None):
    if install_name is None:
        install_name = package
    try:
        __import__(package)
        logging.info(f"Imported {package}.")
    except ImportError:
        logging.info(f"Installing {install_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            logging.info(f"Installed {install_name}.")
            __import__(package)
        except subprocess.CalledProcessError as e:
            logging.critical(f"Failed to install {install_name}. Error: {e}")
            sys.exit(1)

required_packages = [
    "websockets", "numpy", "pandas", "sklearn",
    "xgboost", "catboost", "tensorflow", "joblib", "scipy", "tqdm"
]
for package in required_packages:
    install_and_import(package)

install_and_import("tcn", install_name="keras-tcn")

# Now import the installed libraries
import websockets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.stats import linregress
import xgboost
import catboost
import keras
from tcn import TCN
from tcn.tcn import ResidualBlock

# --- CONFIG ---
SYMBOL = "R_100"
DATABASE_PATH = "tick_data.db"
MODEL_DIR = "models"
MIN_TRAINING_SAMPLES = 500 # Minimum number of ticks before first training
MODEL_UPDATE_INTERVAL = 3600 # Retrain models every hour (3600 seconds)
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("trainer_bot_rise_fall.log"),
                        logging.StreamHandler()
                    ])

# --- Data Fetching and Preprocessing ---
def fetch_data_from_db(limit=None):
    """Fetches clean data from the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    query = f"SELECT quote, epoch FROM ticks WHERE symbol = '{SYMBOL}' ORDER BY epoch"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_features_and_labels(df: pd.DataFrame, window_size: int, duration_ticks: int) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Transforms raw tick data into a rich feature set and labels.
    The most critical part of our analytical engine.
    """
    if len(df) < window_size + duration_ticks:
        return None, None, None

    features_list = []
    labels_list = []

    # Use a loop for feature creation to ensure alignment
    for i in tqdm(range(len(df) - window_size - duration_ticks), desc="Creating Dataset"):
        window = df.iloc[i:i+window_size]['quote'].values
        target_price = df.iloc[i+window_size+duration_ticks-1]['quote']
        entry_price = window[-1]

        # --- Features from the window ---
        features = {}
        # Volatility
        features['volatility_20'] = np.std(window[-20:])
        features['volatility_50'] = np.std(window[-50:])
        features['volatility_100'] = np.std(window[-100:])
        
        # Trend and Momentum
        slope, _, r_value, _, _ = linregress(range(window_size), window)
        features['trend_slope'] = slope
        features['trend_r_value'] = r_value
        features['mean_reversion_score'] = window[-1] - np.mean(window)
        
        # Rate of Change
        features['roc_20'] = (window[-1] - window[-20]) / window[-20] if window[-20] != 0 else 0
        features['roc_50'] = (window[-1] - window[-50]) / window[-50] if window[-50] != 0 else 0
        
        # Moving Average Crossovers
        ma_short = np.mean(window[-20:])
        ma_long = np.mean(window[-50:])
        features['ma_crossover'] = ma_short - ma_long
        
        # Lagged prices
        for j in range(1, 11):
            features[f'lag_price_{j}'] = window[-j]
        
        features_list.append(features)
        
        # --- Label: Rise (1) or Fall (0) ---
        label = 1 if target_price > entry_price else 0
        labels_list.append(label)

    # Convert to DataFrame
    X_classical = pd.DataFrame(features_list)
    y = pd.Series(labels_list)

    # For sequential models
    X_seq = np.array([df.iloc[i:i+window_size]['quote'].values for i in range(len(df) - window_size - duration_ticks)])
    X_seq = X_seq.reshape(X_seq.shape[0], window_size, 1)

    return X_classical, y, X_seq

# --- Model Building Functions ---
def build_xgboost_model(X_train, y_train):
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)
    return model

def build_catboost_model(X_train, y_train):
    model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, loss_function='Logloss', verbose=0)
    model.fit(X_train, y_train)
    return model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, activation='relu', input_shape=input_shape, return_sequences=True),
        GRU(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_tcn_model(input_shape):
    model = Sequential([
        TCN(64, activation='relu', input_shape=input_shape, return_sequences=True),
        TCN(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Training and Saving Logic ---
def train_and_save_models():
    """Fetches data, trains all models, and saves them to disk."""
    logging.info("Starting model training process...")

    # Fetch data from the database
    df = fetch_data_from_db()
    if len(df) < MIN_TRAINING_SAMPLES:
        logging.info(f"Not enough data to train. Need {MIN_TRAINING_SAMPLES}, have {len(df)}.")
        return

    # Create feature sets
    X_classical, y, X_seq = create_features_and_labels(df, window_size=200, duration_ticks=10)
    if X_classical is None:
        logging.info("Not enough data after feature creation. Skipping training.")
        return
        
    # Split data
    split_point = int(len(X_classical) * 0.8)
    X_train_c, X_test_c = X_classical[:split_point], X_classical[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    X_train_s, X_test_s = X_seq[:split_point], X_seq[split_point:]

    # Train and save classical models
    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)
    
    models_to_train = {
        'XGBoost': (build_xgboost_model, X_train_c_scaled, y_train, X_test_c_scaled, y_test, True, scaler),
        'CatBoost': (build_catboost_model, X_train_c, y_train, X_test_c, y_test, True, None),
        'GRU': (build_gru_model, X_train_s, y_train, X_test_s, y_test, False, None),
        'TCN': (build_tcn_model, X_train_s, y_train, X_test_s, y_test, False, None),
        'CNN': (build_cnn_model, X_train_s, y_train, X_test_s, y_test, False, None),
    }

    for name, (builder, X_train, y_train, X_test, y_test, is_classical, scaler) in models_to_train.items():
        logging.info(f"Training {name} model...")
        model = builder(X_train.shape[1:]) if not is_classical else builder(X_train, y_train)

        if not is_classical:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                # CORRECTION START
                ModelCheckpoint(filepath=f"{MODEL_DIR}/{name}_best.keras", monitor='val_loss', save_best_only=True)
            ]
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
            
            # Use the best model saved by the checkpoint
            model = load_model(f"{MODEL_DIR}/{name}_best.keras")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # Save the final model with accuracy in the name
            model.save(f"{MODEL_DIR}/{name}_acc{accuracy:.4f}.keras")
            
            logging.info(f"{name} model trained. Accuracy: {accuracy:.2%}")
        else:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, np.round(y_pred))
            
            # Save the model and scaler in .pkl format for speed
            model_data = {
                'name': name,
                'model': model,
                'accuracy': accuracy,
            }
            if scaler:
                model_data['scaler'] = scaler
            
            with open(f"{MODEL_DIR}/{name}_acc{accuracy:.4f}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"{name} model trained. Accuracy: {accuracy:.2%}")

    logging.info("Model training complete. New models are available.")

# --- The Main Trainer Loop ---
async def run_trainer_loop(queue: Queue):
    """The main loop for the trainer process."""
    last_training_time = 0.0
    
    while True:
        # Get data from the main process
        try:
            # We don't need a queue as the main process is writing to a shared database
            pass
        except:
            pass

        # Check for training conditions
        conn = sqlite3.connect(DATABASE_PATH)
        tick_count = pd.read_sql_query(f"SELECT COUNT(*) FROM ticks WHERE symbol = '{SYMBOL}'", conn).iloc[0, 0]
        conn.close()

        if tick_count >= MIN_TRAINING_SAMPLES and (time.time() - last_training_time) >= MODEL_UPDATE_INTERVAL:
            logging.info(f"Training conditions met. Tick count: {tick_count}. Time since last train: {time.time() - last_training_time:.2f}s.")
            train_and_save_models()
            last_training_time = time.time()
        
        await asyncio.sleep(60) # Check every 60 seconds

if __name__ == "__main__":
    asyncio.run(run_trainer_loop(None))