import requests
import pandas as pd
import pika
from sklearn.datasets import load_iris

#
def post_data(data, queue_name, host="localhost", port=5672, user="guest",
              password="guest"):
  connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, port=port,
                              credentials=pika.PlainCredentials(user,
                                                                password)))
  channel = connection.channel()
  channel.queue_declare(queue=queue_name, durable=True)
  channel.basic_publish(exchange='', routing_key=queue_name,
                        body=data.encode('utf-8'))
  connection.close()
  print(f"Sent to queue: {queue_name}, message: %r" % data)


if __name__ == "__main__":
  url = "http://localhost:8000"
  run_id = "1389fa9a1c0f46c78eb1b351e6703991"
  resp = requests.get(url + "/model/" + run_id)
  print(resp.content)
  print(resp)
  #
  df = pd.read_csv("../models/iris_random_60_samples.csv",sep=";")
  print(f"df: \n{df.tail(5)}")
  print(f"df columns: {list(df.columns)}")
  # Convert to JSON
  data_json = df.to_json(
    orient="records")  # JSON-t rekordok listájaként alakítja át

  print(f"data json: \n{data_json}")

  post_data(data_json, "iris")
  resp = requests.get(url + "/predict/iris")

  ## send this to the frontend.

  print(resp.content)
