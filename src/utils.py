import os
import random
import numpy as np
import torch
import pickle


def seed_everything(seed=42):
    """Set all random seeds for reproducibility. Objects like DataLoader are not seeded, for simplicity
    and to avoid over-complexity."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if getattr(torch.backends, 'mps', None):
        torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    ''' The loss, ref to the paper and its code.'''

    return float(np.sqrt(((predictions - targets) ** 2).mean()))


# For general and CO2 scalers
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_scaler_path = os.path.join(_project_root, "saved_models", "scalers.pkl")


def save_scalers(general_scaler, conc_scaler, path=default_scaler_path):
    """Save fitted MinMaxScalers to file."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:

        pickle.dump({"general_scaler": general_scaler, "conc_scaler": conc_scaler}, f)


def load_scalers(path=default_scaler_path):
    """Load fitted MinMaxScalers from file."""

    with open(path, "rb") as f:

        scalers = pickle.load(f)
        
    return scalers["general_scaler"], scalers["conc_scaler"]
