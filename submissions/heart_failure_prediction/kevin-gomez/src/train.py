import sklearn
import pandas as pd
import joblib
import pickle
import cloudpickle

from sklearn.ensemble import RandomForestClassifier
import os

current_file = __file__


data_folder = os.path.join(os.path.dirname(current_file), "..", "data")
data_file = os.path.join(data_folder, "heart.csv")
model_path = os.path.join(data_folder, "model", "model.pkl")


model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=3,
)


data = pd.read_csv(data_file)

model.fit(data.drop(columns=["target"]), data["target"])

cloudpickle.dump(model, open(model_path, "wb"))

