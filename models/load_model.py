import mlflow
import pandas as pd
from sklearn.datasets import load_iris
import mlflow.sklearn
from mlflow.entities import ViewType


print("started")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("tracking set")


client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Default")
print(experiment.experiment_id)


runs = client.search_runs(experiment_ids=[experiment.experiment_id])
for run in runs:
    print(f"Run ID: {run.info.run_id}, Status: {run.info.status}, Artifacts URI: {run.info.artifact_uri}")

runs2 = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["end_time DESC"]  # Order runs by start time in descending order
)

run_id = runs2[0].info.run_id
print(f"Run ID: {run_id}")

run_data = client.get_run(run_id)
artifact_uri = run_data.info.artifact_uri

model_uri = f"{artifact_uri}/model"
print(f"model uri: {model_uri}")

model2 =  mlflow.sklearn.load_model(model_uri)

model =  mlflow.sklearn.load_model(f"runs:/{run_id}//model")

print("model loaded")
#
# #%% működik-e?
data = load_iris()
#
print("iris loaded")
#
#
#
client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
print("client loaded")
#

#
# Step 2: Retrieve logged data from the run
run_data_dict = client.get_run(run_id).data.to_dictionary()
print(f"Run data loaded, dict: {run_data_dict}")

# Step 3: Load Iris dataset and extract features using logged column names
X = pd.DataFrame(data.data, columns=data.feature_names)  # Ensure DataFrame with feature names
test_input = X.loc[:, eval(run_data_dict["params"]["input"])]  # Use logged input columns

# Step 4: Predict using the model
predictions = model.predict(test_input)
print("Predictions for the dataset:")
print(predictions)
