import numpy as np
import pandas as pd

from src.models.skforecast.ensemble_forecaster import EnsembleForecaster


def test_ensemble_smoke_train_predict():
    # small synthetic series to validate training / predict shape
    n = 100
    y = pd.Series(np.sin(np.linspace(0, 10, n)) + np.random.normal(scale=0.01, size=n))

    lags = 5
    steps = 5
    # split
    split = 80
    y_train = y.iloc[:split]
    y_val = y.iloc[split:]

    forecaster = EnsembleForecaster(
        lags=lags, steps=steps, base_regressors=["ridge", "ridge"]
    )  # lightweight
    forecaster.fit(y_train, validation=(y_val, None), save_artifacts=False)

    last_window = y_train.values[-lags:]
    preds = forecaster.predict(last_window)
    assert preds.shape == (steps,)
    assert not np.any(np.isnan(preds))
