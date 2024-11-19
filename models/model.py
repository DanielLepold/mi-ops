import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Create input example just for explanation of the model
input_example = X_train.iloc[:5]  # Take a small sample as input example

# Infer signature
signature = infer_signature(input_example, model.predict(X_train))

print("MfFlow adjustments")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient("http://127.0.0.1:5000")

name = "Default"

try:
  client.create_experiment(name)
except Exception as e:
  pass

experiment_id = client.get_experiment_by_name(name).experiment_id

with mlflow.start_run(experiment_id=0) as run:
  run_id = mlflow.active_run().info.run_id
  mlflow.sklearn.log_model(model, "model")
  mlflow.log_param("input", X.columns.tolist())
  mlflow.log_param("signature", signature)

  # Save y_test as CSV
  y_test_df = pd.DataFrame({"y_test": y_test})
  y_test_file = "y_test.csv"
  y_test_df.to_csv(y_test_file, index=False)

  # Log y_test as artifact
  mlflow.log_artifact(y_test_file)



print("MfFlow finished")

