# Advanced Model Implementation Plan (Detailed Tasks)

## Overview

This document outlines the strategic plan for enhancing the StockPredictor V1 system by implementing advanced time-series and hybrid models. The goal is to improve prediction accuracy for the 10-day and 30-day horizons by better capturing the temporal dynamics of financial data.

The plan has been broken down into a series of dependent tasks to provide a clear development roadmap.

---

## Detailed Task Breakdown

### Task 1: Implement Data Stationarity Pipeline
*   **ID**: `c8195ef6-e550-4ca8-ade1-a9af95b047c8`
*   **Status**: Done
*   **Description**: Create a reusable module to test for and apply stationarity transformations (differencing, pct change) to time-series data. This module is a foundational requirement for all subsequent modeling tasks.
*   **Implementation Guide**:
    1.  Add `statsmodels` to `pyproject.toml`.
    2.  Create `src/data_utils/stationarity_transformer.py`.
    3.  Inside, implement `transform_to_stationary(series)` which uses the `adfuller` test and returns the best transformed series and the transformation name.
    4.  Also implement `inverse_transform_series(series, transform_name, original_series)` to revert predictions.
    5.  Modify `prepare_ml_data_for_training_with_cleaning` in `src/data_utils/ml_data_pipeline.py` to use `transform_to_stationary` on the target and features, and to return a `transformation_manifest` dictionary.
*   **Verification Criteria**: The data pipeline should output a DataFrame with stationary features and a manifest detailing the transformations applied. Unit tests should validate the transformation and inverse transformation logic.
*   **Dependencies**: None

### Task 2: Create PyTorch Model Base Class
*   **ID**: `cb6f2272-2f81-40f3-be1b-30033f0738d4`
*   **Status**: Done
*   **Description**: Abstract the common logic for PyTorch-based models (LSTM, Transformer) into a reusable base class to avoid code duplication and ensure consistency in training, prediction, and MLflow integration.
*   **Implementation Guide**:
    1.  Create `src/models/time_series/base_pytorch_model.py`.
    2.  Define `PyTorchBasePredictor(BaseModel)`.
    3.  Implement a generic PyTorch training loop within the `fit` method.
    4.  Override `save_model` and `load_model` to use `torch.save`/`torch.load` for the model's `state_dict` and handle MLflow artifacts correctly for PyTorch models.
    5.  Add device management (CPU/GPU).
*   **Verification Criteria**: The base class provides a functional, inheritable structure for training and saving PyTorch models within the project's framework.
*   **Dependencies**: None

### Task 3: Implement ARIMA Baseline Model
*   **ID**: `70cfbc27-3750-4880-9e30-6cb745fcb2ee`
*   **Status**: Done
*   **Description**: Develop and integrate a classical ARIMA model to serve as a statistical baseline for performance comparison.
*   **Implementation Guide**:
    1.  Add `pmdarima` to `pyproject.toml`.
    2.  Create `src/models/time_series/arima_model.py`.
    3.  Define `ARIMAModel(BaseModel)` class.
    4.  Implement `fit` using `pmdarima.auto_arima` on the stationary data.
    5.  Implement `predict` and use the `inverse_transform_series` utility function with the `transformation_manifest` to get predictions in the original scale.
    6.  Update `src/main_pipeline_runner.py` or create a new training script to run and evaluate the `ARIMAModel`.
*   **Verification Criteria**: The ARIMA model can be trained and evaluated within the main pipeline, and its results are logged to MLflow. Predictions are correctly inverse-transformed.
*   **Dependencies**: `Implement Data Stationarity Pipeline`

### Task 4: Implement LSTM Time Series Model
*   **ID**: `f96db412-0b21-4c96-8eb7-48dfa426ebb2`
*   **Description**: Develop a Recurrent Neural Network (LSTM) model to capture complex temporal patterns from the stationary time-series data.
*   **Implementation Guide**:
    1.  Create `src/data_utils/sequential_data_loader.py` with a `TimeSeriesDataset(torch.utils.data.Dataset)` to create sequence windows from the stationary data.
    2.  Create `src/models/time_series/lstm_model.py`.
    3.  Inside, define the network architecture `LSTMModule(torch.nn.Module)`.
    4.  Define the predictor class `LSTMPredictor(PyTorchBasePredictor)`. Its `_create_model` method will instantiate `LSTMModule`.
    5.  Create a new training script `src/train_lstm_model.py` to run and evaluate the `LSTMPredictor`.
*   **Verification Criteria**: The LSTM model can be trained and evaluated. Its architecture is defined, and it correctly uses the sequential data loader and the PyTorch base class. Results are logged to MLflow.
*   **Dependencies**: `Implement Data Stationarity Pipeline`, `Create PyTorch Model Base Class`

### Task 5: Implement Hybrid XGBoost-LSTM Model
*   **ID**: `72906d8e-d45d-4855-af9d-bee00f53b3b7`
*   **Description**: Combine the predictions from the LSTM model as an additional feature for the powerful XGBoost model to create a hybrid model.
*   **Implementation Guide**:
    1.  Modify `prepare_ml_data_for_training_with_cleaning` to accept an optional `extra_features` DataFrame to be concatenated with the main feature set `X`.
    2.  Create `src/train_hybrid_model.py`.
    3.  The script will first load and train the `LSTMPredictor`.
    4.  Then, it will generate predictions from the LSTM model.
    5.  It will then call the data preparation pipeline, passing the LSTM predictions as `extra_features`.
    6.  Finally, it will train and evaluate the standard `XGBoostModel` on this enriched dataset.
*   **Verification Criteria**: The hybrid model pipeline can be executed, and the performance of the XGBoost model with the added LSTM feature is logged and can be compared to the baseline XGBoost.
*   **Dependencies**: `Implement LSTM Time Series Model`

### Task 6: Implement Transformer Time Series Model
*   **ID**: `896f95ce-19c4-4c1f-96a7-162fd142fb9d`
*   **Description**: Explore the state-of-the-art Transformer architecture for time-series forecasting.
*   **Implementation Guide**:
    1.  Create `src/models/time_series/transformer_model.py`.
    2.  Define the network architecture `TransformerModule(torch.nn.Module)`.
    3.  Define the predictor class `TransformerPredictor(PyTorchBasePredictor)`.
    4.  Create a new training script `src/train_transformer_model.py` to run and evaluate the `TransformerPredictor`. The existing `TimeSeriesDataset` can be reused.
*   **Verification Criteria**: The Transformer model can be trained and evaluated, and its results are logged to MLflow for comparison with other models.
*   **Dependencies**: `Implement Data Stationarity Pipeline`, `Create PyTorch Model Base Class`

### Task 7: Comparative Analysis and Reporting
*   **ID**: `6d35063c-d2b3-42e7-ade3-9e495967b1b4`
*   **Description**: Conduct a thorough comparison of all implemented models (ARIMA, XGBoost, LSTM, Hybrid, Transformer) and create a final report with recommendations.
*   **Implementation Guide**:
    1.  Execute all the training scripts for the different models.
    2.  Use the MLflow UI to compare the performance metrics (e.g., RMSE, MAE, Directional Accuracy) across all model runs.
    3.  Create a markdown report (`docs/05-model-training/MODEL_COMPARISON_REPORT.md`) summarizing the findings, including tables and charts of the results.
    4.  Conclude with a recommendation for the best model to be considered for production deployment based on accuracy, training time, and complexity.
*   **Verification Criteria**: A detailed report is produced that clearly compares all models and provides a data-driven recommendation.
*   **Dependencies**: `Implement ARIMA Baseline Model`, `Implement Hybrid XGBoost-LSTM Model`, `Implement Transformer Time Series Model`

---

## Original Implementation Plan (Phases)

This section contains the original high-level plan, preserved for contextual reference.

### Phase 0: Foundational Preprocessing - Data Stationarity (9 hours total)

**Objective**: Create a robust, automated, and reusable pipeline to ensure all time-series data (both target and features) is stationary before being used for model training. This is a critical step for improving model accuracy and generalization.

#### Task 0.1: Develop Automated Stationarity Transformation Module
*   **Assignee Role**: Data Scientist / ML Engineer
*   **Description**: Create a new module (e.g., `src/data_utils/stationarity_transformer.py`). This module should contain a class or function that takes a time series (as a pandas Series) and automatically finds the best stationary transformation.
    *   Implement transformations: percentage change, first difference, and log transformation.
    *   Integrate the Augmented Dickey-Fuller (ADF) statistical test to check for stationarity (p-value < 0.05).
    *   The function should return the transformed, stationary series and the name of the transformation applied. If multiple transformations achieve stationarity, it should select the one with the most negative ADF statistic, as this indicates stronger stationarity.
*   **Deliverables**: A well-documented and tested stationarity transformation module.
*   **Estimated Time**: 6 hours

#### Task 0.2: Integrate Stationarity Module into the Main Data Pipeline
*   **Assignee Role**: ML Engineer
*   **Description**: Modify the primary data preparation pipeline (`prepare_ml_data_for_training_with_cleaning`) to use the new stationarity module.
    *   Apply the stationarity transformation to the target variable (`close` price).
    *   Iterate through all relevant time-series feature columns and apply the transformation to each one.
    *   The pipeline should output the fully preprocessed, stationary DataFrame ready for modeling.
*   **Deliverables**: An updated data preparation pipeline that automatically handles stationarity for all time-series data.
*   **Estimated Time**: 3 hours

---

### Phase 1: ARIMA Baseline Model (15 hours total)

**Objective**: Establish a robust statistical baseline using a classical time-series model on the newly prepared stationary data.

#### Task 1.1: Research and Library Selection
*   **Assignee Role**: ML Engineer
*   **Description**: Research and select the best Python library for ARIMA implementation (e.g., `statsmodels`, `pmdarima`).
*   **Deliverables**: A decision document or updated `pyproject.toml` with the chosen library.
*   **Estimated Time**: 3 hours

#### Task 1.2: Create `ARIMAModel` Base Class
*   **Assignee Role**: ML Engineer
*   **Description**: Create a new file `src/models/time_series/arima_model.py`. Define an `ARIMAModel` class that inherits from `src/models/base_model.py`.
*   **Deliverables**: The `arima_model.py` file with the initial class structure.
*   **Estimated Time**: 3 hours

#### Task 1.3: Implement Data Preparation for ARIMA
*   **Assignee Role**: ML Engineer
*   **Description**: The `ARIMAModel` should now directly consume the stationary time series produced by the **Phase 0** pipeline. This task involves ensuring the model correctly selects the transformed target variable for training.
*   **Deliverables**: A data handling method within the `ARIMAModel` class that correctly uses the pre-processed stationary data.
*   **Estimated Time**: 3 hours

#### Task 1.4: Implement ARIMA Training and Forecasting
*   **Assignee Role**: ML Engineer
*   **Description**: Implement the `fit()` and `predict()` methods. The `fit` method should automatically determine the optimal (p, d, q) parameters. The `predict` method should forecast for the 10-day and 30-day horizons. **Note**: The predictions will be in the transformed space and will need to be inverse-transformed back to absolute price changes or prices for evaluation.
*   **Deliverables**: Functional `fit()` and `predict()` methods, including inverse transformation logic.
*   **Estimated Time**: 3 hours

#### Task 1.5: Integrate ARIMA into Evaluation Pipeline
*   **Assignee Role**: ML Engineer
*   **Description**: Update the main training script to train and evaluate the `ARIMAModel`. Ensure its performance is logged and compared using the established metrics.
*   **Deliverables**: An updated training script that includes the ARIMA model.
*   **Estimated Time**: 3 hours

---

### Phase 2: LSTM/GRU Model Implementation (15 hours total)

**Objective**: Develop a Recurrent Neural Network (RNN) model to capture complex temporal patterns from the stationary time-series features.

#### Task 2.1: Implement Sequential Data Loader for Stationary Data
*   **Assignee Role**: Deep Learning Engineer
*   **Description**: Create a custom PyTorch `Dataset` and `DataLoader` for handling time-series data. The loader should take the stationary feature DataFrame from the **Phase 0** pipeline and create overlapping sequences.
*   **Deliverables**: A `TimeSeriesDataset` class in a new file, e.g., `src/data_utils/sequential_data_loader.py`.
*   **Estimated Time**: 3 hours

#### Task 2.2: Define `LSTMModel` Architecture
*   **Assignee Role**: Deep Learning Engineer
*   **Description**: Create `src/models/time_series/lstm_model.py`. Define a `LSTMModel` class using PyTorch (`torch.nn.Module`). The architecture should take the multi-feature stationary sequences as input.
*   **Deliverables**: The `lstm_model.py` file with the PyTorch model definition.
*   **Estimated Time**: 3 hours

#### Task 2.3: Implement Training Loop for LSTM
*   **Assignee Role**: Deep Learning Engineer
*   **Description**: Implement the `fit()` method for the `LSTMModel`. This will involve a standard PyTorch training loop using the stationary sequential data.
*   **Deliverables**: A functional `fit()` method that can train the LSTM model.
*   **Estimated Time**: 3 hours

#### Task 2.4: Implement Prediction and Evaluation for LSTM
*   **Assignee Role**: Deep Learning Engineer
*   **Description**: Implement the `predict()` method. As with ARIMA, the model's output will be in the transformed (stationary) space and will require an inverse transformation step to be evaluated as a price change.
*   **Deliverables**: A functional `predict()` method with inverse transformation logic.
*   **Estimated Time**: 3 hours

#### Task 2.5: MLflow Integration for LSTM
*   **Assignee Role**: Deep Learning Engineer
*   **Description**: Integrate the `LSTMModel` with MLflow. Log hyperparameters, training/validation loss, and final performance metrics. Save the trained model's `state_dict` as an artifact.
*   **Deliverables**: `save_model` and `load_model` functionality for the `LSTMModel`.
*   **Estimated Time**: 3 hours

---

### Phase 3: Hybrid Model (XGBoost + LSTM) Implementation (9 hours total)

**Objective**: Combine the predictive power of the feature-rich XGBoost model (on stationary tabular features) with the temporal awareness of the LSTM model.

#### Task 3.1: Modify XGBoost Data Pipeline for Hybrid Input
*   **Assignee Role**: ML Engineer
*   **Description**: The main data pipeline from **Phase 0** already produces the stationary features needed for XGBoost. This task involves updating it to also accept the predictions from the trained LSTM model as an additional feature column.
*   **Deliverables**: Updated data preparation function that can merge LSTM predictions.
*   **Estimated Time**: 3 hours

#### Task 3.2: Implement End-to-End Training for Hybrid Model
*   **Assignee Role**: ML Engineer
*   **Description**: Create a new master script that orchestrates the training of the hybrid model. The script should first train the `LSTMModel` on stationary sequential data, generate predictions, and then train the final `XGBoostModel` using the stationary tabular data combined with the LSTM predictions.
*   **Deliverables**: A Python script to run the full hybrid training pipeline.
*   **Estimated Time**: 3 hours

#### Task 3.3: Evaluate and Tune Hybrid Model
*   **Assignee Role**: ML Engineer
*   **Description**: Run the full pipeline and evaluate its performance against the individual models. Log results in MLflow.
*   **Deliverables**: MLflow experiment runs with comparative results.
*   **Estimated Time**: 3 hours

---

### Phase 4: Transformer Model Exploration (Advanced) (9 hours total)

**Objective**: Explore the state-of-the-art Transformer architecture, leveraging the robust stationary data pipeline.

#### Task 4.1: Research and Implement Transformer Architecture
*   **Assignee Role**: Deep Learning Engineer / Researcher
*   **Description**: Research and implement a Transformer-based architecture suitable for multi-feature stationary time-series data. The existing `TimeSeriesDataset` can be adapted for this.
*   **Deliverables**: A `transformer_model.py` file and any necessary updates to the data loader.
*   **Estimated Time**: 3 hours

#### Task 4.2: Train and Evaluate Transformer Model
*   **Assignee Role**: Deep Learning Engineer
*   **Description**: Implement the training and evaluation loop for the Transformer model, ensuring results are inverse-transformed for final evaluation. Log all results to MLflow.
*   **Deliverables**: A training script and MLflow runs for the Transformer model.
*   **Estimated Time**: 3 hours

#### Task 4.3: Final Comparative Analysis
*   **Assignee Role**: Senior ML Engineer / Data Scientist
*   **Description**: Conduct a thorough comparison of all implemented models: ARIMA, XGBoost, LSTM, Hybrid, and Transformer, all trained on the same foundation of stationary data.
*   **Deliverables**: A final report summarizing the findings and recommending the best model for production deployment.
*   **Estimated Time**: 3 hours 