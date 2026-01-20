"""Model training interfaces."""

from .classical import train_xgboost_model

# Lazy imports for torch-dependent modules to avoid scipy/torch import order segfaults.
# These are only loaded when explicitly accessed.
def __getattr__(name: str):
    if name == "LSTMMinutesPredictor":
        from .deep import LSTMMinutesPredictor
        return LSTMMinutesPredictor
    if name == "train_lstm_model":
        from .deep import train_lstm_model
        return train_lstm_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "train_xgboost_model",
    "LSTMMinutesPredictor",
    "train_lstm_model",
]
