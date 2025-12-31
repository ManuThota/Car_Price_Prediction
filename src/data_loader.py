import pandas as pd

def load_data(path : str) -> pd.DataFrame:
    """
    Load Dataset from given path
    """
    return pd.read_csv(path)