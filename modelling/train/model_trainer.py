import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
# Set random state for reproducibility
RANDOM_STATE = int(os.getenv("RANDOM_STATE", ""))

BEST_PARAMS = os.getenv("BEST_PARAMS", "")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "")

class RfModelTrainer():
    def __init__(self, X: np.ndarray, y: pd.Series) -> None:
        self.X = X
        self.y = y
        with open(BEST_PARAMS, "r") as file:
            self.best_params = json.load(file)
    
    def run_model(self):
        self.rf = RandomForestClassifier(**self.best_params, random_state=RANDOM_STATE)
        self.rf.fit(self.X, self.y)

        # Save the model to file
        with open(MODEL_SAVE_PATH, "wb") as model:
            pickle.dump(self.rf, model)

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from modelling.preprocess.data_preprocessor import Preprocessor
    DATA_PATH = os.getenv("DATA_PATH", "")
    pro = Preprocessor(data_path=DATA_PATH)
    data = pro.read_data()
    X_scaled, y = pro.preprocess(data=data)
    trainer = RfModelTrainer(X=X_scaled, y=y)
    trainer.run_model()
