"""Module with utitlity functions."""
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        if np.issubdtype(type(obj), np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
