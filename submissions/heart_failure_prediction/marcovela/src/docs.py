import cloudpickle
import os


current_file = __file__


data_folder = os.path.join(os.path.dirname(current_file), "..", "data")
data_file = os.path.join(data_folder, "heart.csv")
model_path = os.path.join(data_folder, "model", "model.pkl")

model = cloudpickle.load(open(model_path, "rb"))

names = model.feature_names_in_.tolist()


print("Feature names used in the model:")
for name in names:
    print(f"- {name}")


