## FastAPI
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import mlflow
import mlflow.sklearn
import pandas as pd
import pika
import time
import logging
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score


model = None
client = None
signature = None
rabbit_connection = None
channel = None
df_y_original = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  global client, rabbit_connection, channel
  client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
  credentials = pika.PlainCredentials(username="guest", password="guest")
  while rabbit_connection is None:
    try:
      rabbit_connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost", port=5672,
                                  credentials=credentials, heartbeat=0))
    except pika.exceptions.AMQPConnectionError:
      logging.error(
        f"Connection to RabbitMQ failed at localhost:5672. Retrying...")
      time.sleep(0.3)
  channel = rabbit_connection.channel()
  channel.basic_qos(prefetch_count=1)

  yield
  channel.close()
  rabbit_connection.close()
  return


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
  """Default path. See /docs for more."""
  return "Hello World"
  ## TODO:


@app.get("/model/{run_id}")
def get_mlflow_model(run_id: str):
  global model, client, signature, df_y_original
  mlflow.set_tracking_uri("http://127.0.0.1:5000")
  model = mlflow.sklearn.load_model(f"runs:/{run_id}//model")
  y_original = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/y_test.csv")
  print(f"y original: {y_original}")

  df_y_original = pd.read_csv(y_original)
  print(f"dataframe y original: \n {df_y_original}")
  run_data_dict = client.get_run(run_id).data.to_dictionary()
  print(run_data_dict)
  signature = eval(run_data_dict["params"]["input"])
  # Download y_test artifact and load into DataFrame
  return signature


@app.get("/predict/{queue}")
async def predict(queue):
  global channel
  method_frame, header_frame, body = channel.basic_get(queue)
  data = body.decode("utf-8")
  print(f"Queue name: {queue}")
  print(f"Delivery tag: {method_frame.delivery_tag}")
  channel.basic_ack(method_frame.delivery_tag)
  X = pd.read_json(data)

  print(f"X columns: {list(X.columns)}")
  # Predictions
  y_pred = model.predict(X)
  print(f"Predictions: {y_pred}")
  X["y_pred"] = y_pred

  print(f"Df_y_original: {df_y_original}")
  # Assuming df_y_original contains the true labels
  if "y_test" not in df_y_original.columns:
    return {"error": "Original y_test data not found.."}


  y_true = df_y_original["y_test"].values

  # Calculate metrics
  precision = precision_score(y_true, y_pred, average="weighted")
  accuracy = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average="weighted")
  recall = recall_score(y_true, y_pred, average="weighted")

  metrics = {
    "precision": precision,
    "accuracy": accuracy,
    "f1_score": f1,
    "recall": recall
  }

  print(f"metrics: {metrics}")

  return {"predictions": X.to_json(orient="records"), "metrics": metrics}

if __name__ == "__main__":
  import uvicorn

  uvicorn.run("server:app", host="localhost", port=8000, reload=True)
