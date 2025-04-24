import pandas as pd

def load_data(path='data/u.data'):
    """Loads MovieLens ratings data from u.data file."""
    ratings = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return ratings
