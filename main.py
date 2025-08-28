# main_rise_fall.py
# The Final Blueprint: A production-grade autonomous trading system.
# It handles real-time data, dynamic risk management, and manages the
# background analytical engine for model training.

import asyncio
import websockets
import json
import os
import sys
import re
import sqlite3
import pandas as pd
import numpy as np
import logging
import joblib
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional, Tuple
from collections import deque, defaultdict
import subprocess
from decimal import Decimal, getcontext
import pickle
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# Set precision for Decimal operations
getcontext().prec = 20

# --- Self-Sufficiency and Dependency Management ---
# This ensures a flawless setup in any environment by installing
# dependencies before anything else. No excuses.
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
from tcn import TCN
from tcn.tcn import ResidualBlock # This is now for completeness
from scipy.stats import linregress
import xgboost
import catboost
from tensorflow.keras.models import load_model # Ensure this is imported for loading .keras models
import keras # Added to handle Keras-related functions if needed.

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("main_bot_rise_fall.log"),
                        logging.StreamHandler()
                    ])

# --- CONFIG ---
API_TOKEN = os.getenv("DERIV_API_TOKEN", "YUa7FW6khNWyW") # Use environment variable for security
if API_TOKEN == "YOUR_API_TOKEN":
    logging.warning("API Token not set as environment variable. Using default.")

SYMBOL = "R_100"
TRADE_DURATION_TICKS = 10 # Predict over the next 10 ticks
MIN_STAKE = 0.35 # Minimum stake to ensure profitability
MIN_DATA_BUFFER = 200 # Minimum historical data to generate features
MODEL_DIR = "models"
DATABASE_PATH = "tick_data.db"
MODEL_COMM_QUEUE = None
BACKGROUND_PROCESS = None
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Database & Data Handling ---
def setup_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quote REAL NOT NULL,
            epoch INTEGER NOT NULL,
            request_time REAL NOT NULL,
            UNIQUE(symbol, epoch) ON CONFLICT IGNORE
        )
    ''')
    conn.commit()
    conn.close()

def parse_tick_price(quote: Any) -> float:
    """Handles both float and integer prices, ensuring consistency."""
    if isinstance(quote, (int, float, Decimal)):
        return float(quote)
    try:
        return float(re.sub(r'[^\d.]', '', str(quote)))
    except (ValueError, TypeError):
        logging.error(f"Failed to parse quote: {quote}. Defaulting to 0.0.")
        return 0.0

class TradingBrain:
    """The mind of the operation. Manages state, logic, and models."""
    def __init__(self):
        self.account_balance: float = 0.0
        self.initial_balance: float = 0.0
        self.daily_profit: float = 0.0
        self.total_trades: int = 0
        self.trading_active: bool = True
        self.historical_prices: deque = deque(maxlen=MIN_DATA_BUFFER)
        self.active_models: Dict[str, Any] = {}
        self.model_accuracies: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.current_stake: float = MIN_STAKE
        self.win_rate: float = 0.0
        
    def generate_features_and_sequences(self, prices: deque) -> Tuple[Any, Any]:
        """
        Creates a rich feature set from raw prices for both classical and sequential models.
        This is our computational edge.
        """
        prices_list = list(prices)
        if len(prices_list) < MIN_DATA_BUFFER:
            return None, None

        window = prices_list[-MIN_DATA_BUFFER:]
        
        # --- Advanced Classical Features ---
        features = {}
        
        # Volatility features
        features['volatility_20'] = np.std(window[-20:])
        features['volatility_50'] = np.std(window[-50:])
        features['volatility_100'] = np.std(window[-100:])
        
        # Trend and Momentum
        slope, intercept, r_value, p_value, std_err = linregress(range(MIN_DATA_BUFFER), window)
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
        
        classical_features = pd.DataFrame([features])
        X_seq = np.array(window).reshape(1, MIN_DATA_BUFFER, 1)

        return classical_features, X_seq
        
    def load_latest_models(self):
        """Loads all available models from the model directory."""
        self.active_models = {}
        self.model_accuracies = {}
        try:
            # Check for both .pkl and .keras files
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') or f.endswith('.keras')]
            if not model_files:
                logging.warning("No models found. Bot will not trade.")
                return

            for filename in tqdm(model_files, desc="Loading Models"):
                file_path = os.path.join(MODEL_DIR, filename)
                try:
                    if filename.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            model_data = pickle.load(f)
                        self.active_models[model_data['name']] = model_data['model']
                        self.model_accuracies[model_data['name']] = model_data['accuracy']
                        if 'scaler' in model_data and model_data['name'] == 'XGBoost':
                             self.scaler = model_data['scaler']
                    # Load the new .keras format
                    elif filename.endswith('.keras'):
                        from tensorflow.keras.models import load_model
                        model = load_model(file_path)
                        
                        name_match = re.match(r'^(.*?)_acc(\d+\.\d+)\.keras$', filename)
                        if name_match:
                            name, acc = name_match.groups()
                            self.active_models[name] = model
                            self.model_accuracies[name] = float(acc)
                except Exception as e:
                    logging.error(f"Failed to load model {filename}: {e}")
        except Exception as e:
            logging.error(f"Error listing models: {e}")

    def get_prediction(self) -> Tuple[str, float]:
        """Ensemble prediction from all loaded models."""
        if not self.active_models or len(self.historical_prices) < MIN_DATA_BUFFER:
            return "NONE", 0.0

        try:
            classical_features, X_seq = self.generate_features_and_sequences(self.historical_prices)
            if classical_features is None or X_seq is None:
                return "NONE", 0.0
                
            ensemble_probabilities = defaultdict(float)
            total_weight = 0.0

            for model_name, model in self.active_models.items():
                if model_name in self.model_accuracies:
                    weight = self.model_accuracies[model_name]
                    if weight < 0.51: continue
                    total_weight += weight

                    if model_name in ['XGBoost', 'CatBoost']:
                        if self.scaler:
                            X_classical_scaled = self.scaler.transform(classical_features)
                            proba = model.predict_proba(X_classical_scaled)[0]
                            call_proba = proba[1]
                        else:
                            continue
                    elif model_name in ['GRU', 'TCN', 'CNN']:
                        proba = model.predict(X_seq, verbose=0)[0]
                        call_proba = proba[0]
                    else:
                        continue
                    
                    ensemble_probabilities['CALL'] += call_proba * weight
                    ensemble_probabilities['PUT'] += (1 - call_proba) * weight
            
            if total_weight == 0.0:
                return "NONE", 0.0

            ensemble_probabilities['CALL'] /= total_weight
            ensemble_probabilities['PUT'] /= total_weight
            
            call_confidence = ensemble_probabilities['CALL']
            put_confidence = ensemble_probabilities['PUT']
            
            if call_confidence > put_confidence:
                return "CALL", call_confidence
            else:
                return "PUT", put_confidence

        except Exception as e:
            logging.error(f"PREDICTION ERROR: {e}")
            return "NONE", 0.0
            
    def get_dynamic_confidence_level(self) -> float:
        """Dynamically adjusts confidence threshold based on recent performance."""
        base_threshold = 0.65 # A high baseline for safety
        
        # Adjust based on win rate
        if self.win_rate > 0.6:
            adjustment = (self.win_rate - 0.6) * 0.2 # Be more aggressive with a higher win rate
        else:
            adjustment = (self.win_rate - 0.6) * 0.5 # Be more conservative with a lower win rate
            
        # Adjust based on consecutive losses
        loss_adjustment = self.consecutive_losses * 0.02 # Increase threshold significantly
        
        # Capped adjustments
        final_threshold = base_threshold + adjustment + loss_adjustment
        return max(0.51, min(0.9, final_threshold)) # Ensure it's between 51% and 90%

    def get_dynamic_stake(self, confidence: float) -> float:
        """Dynamically adjusts stake based on confidence and Martingale."""
        martingale_stake = MIN_STAKE * (1.8 ** self.consecutive_losses) # Martingale multiplier
        confidence_multiplier = (confidence - 0.5) * 2 # 0.5 confidence -> 0, 1.0 confidence -> 1.0
        
        base_stake = MIN_STAKE + (self.account_balance * 0.01 * confidence_multiplier)
        
        final_stake = max(martingale_stake, base_stake)
        
        # Capital Protection: Never risk more than 10% of the account balance
        stake_cap = self.account_balance * 0.1
        final_stake = min(final_stake, stake_cap)
        
        return round(final_stake, 2)

    def manage_risk_protocol(self):
        """Monitors drawdown and consecutive losses to protect capital."""
        # Dynamic Drawdown Limit
        current_drawdown = self.initial_balance - self.account_balance
        if current_drawdown > self.initial_balance * 0.2: # 20% drawdown limit
            self.trading_active = False
            logging.critical("🚨 DRAWDOWN LIMIT REACHED. TRADING PAUSED.")
            return

        # Consecutive Loss Limit
        if self.consecutive_losses >= 10:
            self.trading_active = False
            logging.critical("🚨 MAX CONSECUTIVE LOSSES REACHED. TRADING PAUSED.")
            return

    def update_martingale_stake(self, win: bool):
        """Updates consecutive win/loss counters."""
        if win:
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

def start_model_trainer_process(queue: Queue):
    """
    Function to be executed by the child process.
    This is where the analytical engine resides.
    """
    try:
        from model_trainer import run_trainer_loop
        asyncio.run(run_trainer_loop(queue))
    except ImportError as e:
        logging.critical(f"Failed to import model_trainer.py. Ensure the file exists: {e}")
    except Exception as e:
        logging.critical(f"🔥🔥 CRITICAL ERROR in trainer process: {e}🔥🔥")
            
async def run_bot():
    """Main execution loop for the bot."""
    global BACKGROUND_PROCESS, MODEL_COMM_QUEUE
    
    setup_database()
    
    # Initialize the communication queue and start the analytical process
    MODEL_COMM_QUEUE = Queue()
    BACKGROUND_PROCESS = Process(target=start_model_trainer_process, args=(MODEL_COMM_QUEUE,))
    BACKGROUND_PROCESS.start()
    logging.info("🧠 Analytical engine (model_trainer.py) started as a background process.")

    brain = TradingBrain()
    brain.load_latest_models()
    
    try:
        logging.info("Connecting to Deriv WebSocket...")
        async with websockets.connect(f"wss://ws.derivws.com/websockets/v3?app_id=85473") as ws:
            # Authentication
            await ws.send(json.dumps({"authorize": API_TOKEN}))
            auth_response = json.loads(await ws.recv())
            if "error" in auth_response:
                logging.critical(f"Authentication Failed! Error: {auth_response['error']['message']}")
                return
            brain.account_balance = float(auth_response["authorize"]["balance"])
            brain.initial_balance = brain.account_balance
            logging.info(f"Authorization successful. Starting balance: ${brain.account_balance:.2f}")

            # --- Critical Startup Logic: Fill in Tick Gaps ---
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            last_epoch_result = cursor.execute("SELECT MAX(epoch) FROM ticks WHERE symbol = ?", (SYMBOL,)).fetchone()
            last_epoch = last_epoch_result[0] if last_epoch_result[0] else None
            
            if last_epoch:
                logging.info(f"Last recorded tick epoch: {last_epoch}. Requesting historical ticks to fill gaps...")
                ticks_history_req = {
                    "ticks_history": SYMBOL,
                    "end": "latest",
                    "start": last_epoch,
                    "style": "ticks"
                }
                await ws.send(json.dumps(ticks_history_req))
                history_response = json.loads(await ws.recv())
                if history_response.get('history'):
                    for tick, epoch in zip(history_response['history']['prices'], history_response['history']['times']):
                        cursor.execute("INSERT OR IGNORE INTO ticks (symbol, quote, epoch, request_time) VALUES (?, ?, ?, ?)",
                                       (SYMBOL, tick, epoch, time.time()))
                    conn.commit()
                    logging.info("Historical tick data filled.")
            conn.close()

            # Subscribe to R_100 ticks
            await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
            await asyncio.sleep(1)
            
            logging.info("Bot is active, awaiting trade opportunities.")
            
            while True:
                response = json.loads(await ws.recv())
                if response.get("msg_type") == "tick":
                    tick = response['tick']
                    price = parse_tick_price(tick['quote'])
                    
                    conn = sqlite3.connect(DATABASE_PATH)
                    cursor = conn.cursor()
                    cursor.execute("INSERT OR IGNORE INTO ticks (symbol, quote, epoch, request_time) VALUES (?, ?, ?, ?)",
                                   (SYMBOL, price, tick['epoch'], time.time()))
                    conn.commit()
                    conn.close()
                    
                    brain.historical_prices.append(price)
                else:
                    logging.debug(f"Received non-tick message type: {response.get('msg_type')}.")
                    continue

                if len(brain.historical_prices) < MIN_DATA_BUFFER:
                    logging.info("Building historical price buffer...")
                    continue

                # --- Core Trading Logic ---
                predicted_type, confidence = brain.get_prediction()
                dynamic_confidence = brain.get_dynamic_confidence_level()

                if confidence < dynamic_confidence:
                    logging.info(f"Prediction confidence ({confidence:.2%}) too low. Skipping trade. Min: {dynamic_confidence:.2%}")
                    continue
                
                brain.manage_risk_protocol()
                if not brain.trading_active:
                    await asyncio.sleep(60) # Pause for 60 seconds before rechecking
                    continue

                stake_to_use = brain.get_dynamic_stake(confidence)
                
                logging.info(f"🔮 Attempting trade: Pred={predicted_type} (Conf: {confidence:.2%}), Stake=${stake_to_use:.2f}")

                try:
                    proposal_req = {
                        "proposal": 1, "amount": stake_to_use, "basis": "stake",
                        "contract_type": "CALL" if predicted_type == "CALL" else "PUT", 
                        "currency": "USD", "duration": TRADE_DURATION_TICKS,
                        "duration_unit": "t", "symbol": SYMBOL
                    }
                    
                    await ws.send(json.dumps(proposal_req))
                    proposal = json.loads(await ws.recv())
                    if "error" in proposal:
                        logging.error(f"Proposal Error: {proposal['error']['message']}. Skipping.")
                        brain.update_martingale_stake(win=False) # Assume loss for safety
                        continue

                    buy_id = proposal["proposal"]["id"]
                    await ws.send(json.dumps({"buy": buy_id, "price": stake_to_use}))
                    buy_response = json.loads(await ws.recv())
                    if "error" in buy_response:
                        logging.error(f"Buy Error: {buy_response['error']['message']}. Skipping.")
                        brain.update_martingale_stake(win=False)
                        continue
                except Exception as e:
                    logging.error(f"API Call Error: {e}. Skipping this trade.")
                    brain.update_martingale_stake(win=False)
                    continue

                try:
                    open_contract_req = {"proposal_open_contract": 1, "contract_id": buy_response['buy']['contract_id'], "subscribe": 1}
                    await ws.send(json.dumps(open_contract_req))
                    while True:
                        contract_status = json.loads(await ws.recv())
                        if contract_status.get('proposal_open_contract', {}).get('is_expired', 0) == 1:
                            break
                        if contract_status.get('proposal_open_contract', {}).get('status') == 'closed':
                            break
                        await asyncio.sleep(0.1)

                except asyncio.TimeoutError:
                    logging.error("Timeout waiting for contract to close.")
                    brain.update_martingale_stake(win=False)
                    continue
                except Exception as e:
                    logging.error(f"Error monitoring contract: {e}")
                    brain.update_martingale_stake(win=False)
                    continue
                
                final_status = contract_status['proposal_open_contract']
                win = final_status.get('profit') is not None and final_status.get('profit', 0) > 0
                profit_amount = float(final_status.get('profit', 0))
                
                # Update stats
                brain.update_martingale_stake(win)
                brain.account_balance += profit_amount
                brain.daily_profit += profit_amount
                brain.total_trades += 1
                brain.win_rate = brain.consecutive_wins / brain.total_trades if brain.total_trades > 0 else 0
                
                outcome_str = "WIN" if win else "LOSS"
                logging.info(f"✅ Trade Result: {outcome_str}. P/L: {profit_amount:+.2f}. Consecutive Losses: {brain.consecutive_losses}. Total Trades: {brain.total_trades}")
                logging.info(f"📊 Daily P/L: ${brain.daily_profit:+.2f}. Current Stake: ${brain.current_stake:.2f}")

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.info("🚨 Bot process cancelled. Shutting down analytical engine...")
    except Exception as e:
        logging.critical(f"🔥🔥 CRITICAL ERROR: {e}🔥🔥")
    finally:
        if BACKGROUND_PROCESS:
            BACKGROUND_PROCESS.terminate()
            logging.info("✅ Analytical engine shut down gracefully.")

def start_model_trainer_process(queue: Queue):
    """
    Function to be executed by the child process.
    This is where the analytical engine resides.
    """
    try:
        from model_trainer import run_trainer_loop
        asyncio.run(run_trainer_loop(queue))
    except ImportError as e:
        logging.critical(f"Failed to import model_trainer.py. Ensure the file exists: {e}")
    except Exception as e:
        logging.critical(f"🔥🔥 CRITICAL ERROR in trainer process: {e}🔥🔥")
    
if __name__ == "__main__":

    asyncio.run(run_bot())
