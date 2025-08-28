
The Analytical Engine: A Production-Grade Trading Bot 🤖

This project is a high-frequency, autonomous trading system designed to predict the last digit of a synthetic index's price tick. It uses a sophisticated dual-process architecture to separate real-time trading from intensive, background model training, ensuring optimal performance, responsiveness, and a self-sufficient setup.

✨ Key Features
Production-Grade Architecture: The system is built for reliability, with a main process (main.py) handling all trading logic, and a separate analytical engine (model_trainer.py) dedicated to continuous, background model training. This architecture prevents trading interruptions during model updates.

Database-Driven Intelligence: Unlike traditional setups, this system uses an SQLite3 database to manage tick data and share models between the main trading process and the analytical engine, ensuring seamless and efficient communication.

Multi-Model Ensemble: The bot is equipped to train and utilize an ensemble of both classical and deep learning models, including XGBoost, CatBoost, GRU, TCN, and CNN.

Dynamic Trading Strategy: Trades are executed only when the ensemble model's confidence in its prediction exceeds a predefined threshold.

Robust Risk Management: Built-in features include daily loss and profit limits, along with a consecutive loss counter to prevent significant drawdowns.

Self-Sufficient Setup: The script is designed to automatically install all necessary Python libraries upon the first run, making it easy to set up in any environment.

💻 How It Works
The system is split into two core components that operate in parallel:

1. main.py (The Executive Brain 🧠)
This is the bot's central control unit. Its responsibilities include:

Connecting to the Deriv WebSocket API.

Executing trades and managing the bot's risk in real-time.

Starting model_trainer.py as a separate, non-blocking process.

Loading the best-performing models from the database for live predictions.

2. model_trainer.py (The Analytical Engine 🔬)
This process runs in the background and is the intelligence behind the system. It handles:

Continuously ingesting tick data from the WebSocket and storing it in the database.

Periodically generating features and sequences from the collected data.

Training and evaluating all the machine learning models.

Saving the newly trained, best-performing models to the database for use by the main process.

🚀 Getting Started
Prerequisites
A Deriv API token https://api.deriv.com/.

Python 3.7 or higher.

Installation
Clone this repository:

Navigate to the project directory:
cd your-repo-name

Update the API_TOKEN variable in main.py with your personal token.

Run the main script:
python main.py

The script will handle the rest, automatically installing all required dependencies and starting the trading bot.

☁️ Recommended: Run on Google Colab
For a significant performance boost, we highly recommend running this script on Google Colab. The enhanced computing power, particularly the dedicated GPU access, will dramatically speed up the training of the deep learning models (GRU, TCN, and CNN), leading to faster and more frequent model updates. Simply upload both main.py and model_trainer.py to your session and run main.py in a code cell.
