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
  run_id = "6981f49373dc407ca81bf36dd6cf4908"
  resp = requests.get(url + "/model/" + run_id)
  print(resp.content)
  print(resp)
  #
  data = load_iris()
  df = pd.DataFrame(data.data, columns=data.feature_names)

  # Convert to JSON
  data_json = df.to_json(
    orient="records")  # JSON-t rekordok listájaként alakítja át

  post_data(data_json, "iris")
  resp = requests.get(url + "/predict/iris")

  ## send this to the frontend.

  print(resp.content)
