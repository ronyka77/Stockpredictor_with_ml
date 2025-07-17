#!/usr/bin/env python3
"""
Training Script for the LSTM Time Series Model
"""
from torch.utils.data import DataLoader

from src.utils.logger import get_logger
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.data_utils.sequential_data_loader import TimeSeriesDataset
from src.models.time_series.lstm_model import LSTMPredictor

logger = get_logger(__name__)
experiment_name = "lstm_model"

def prepare_data(sequence_length=30, logger=None):
    logger.info("STEP 1: Preparing data with stationarity transformation...")
    data_dict = prepare_ml_data_for_training_with_cleaning(
        prediction_horizon=10,
        apply_stationarity_transform=True,
        use_cache=False
    )
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length=sequence_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length=sequence_length)
    return data_dict, train_dataset, test_dataset

def save_model(lstm_model, logger):
    logger.info("\nSTEP 4: Saving LSTM model...")
    lstm_model.save_model(experiment_name)
    logger.info(f"✅ Model saved to {experiment_name}")

def run_lstm_training():
    logger.info("="*80)
    logger.info("🚀 STARTING LSTM MODEL TRAINING PIPELINE")
    logger.info("="*80)
    try:
        sequence_length = 20
        batch_size = 512
        data_dict, train_dataset, test_dataset = prepare_data(sequence_length, logger)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        current_prices = X_test['close'].values
        logger.info(f"Data loaders created. Sequence length: {sequence_length}, Batch size: {batch_size}")
        lstm_config = {
            'input_size': X_train.shape[1],
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2,
            'epochs': 10,
            'learning_rate': 0.002,
            'batch_size': batch_size
        }
        lstm_model = LSTMPredictor(config=lstm_config)
        lstm_model.fit(train_loader, val_loader=test_loader, feature_names=X_train.columns.tolist())
        lstm_model.evaluate(X_test, y_test, current_prices=current_prices, confidence_method='leaf_depth')
        logger.info("✅ LSTM model training complete.")

        # evaluate_model(lstm_model, data_dict['X_test'], test_dataset, data_dict, logger, device)
        save_model(lstm_model, logger)
    except Exception as e:
        logger.error(f"❌ An error occurred during the LSTM training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_lstm_training() 