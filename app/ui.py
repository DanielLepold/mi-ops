## Streamlit
import streamlit as st
import pandas as pd
import pika
import requests
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, recall_score, RocCurveDisplay
import matplotlib.pyplot as plt
#%% Simple interface

host = "localhost"
port = 5672
user = "guest"
password = "guest"
url = "http://localhost:8000"
upload = st.file_uploader("Upload CSV.")


run_id = st.text_input("Run ID")
if st.button("Load model") and (run_id is not None or run_id != ""):
    resp = requests.get(url + "/model/" + run_id)
    st.write(resp.content)
else:
    st.write("Click the button to load model.")

resp = requests.get(url + "/model/current")
run_id = resp.content.decode("utf-8")
st.write(f"Current model: {run_id}")

if upload is not None:
    data = pd.read_csv(upload, sep=";" )
    print(f"Data: {data}")

    # Convert to JSON
    data_json = data.to_json(
      orient="records")  # JSON-t rekordok listájaként alakítja át

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=pika.PlainCredentials(user, password)))
    channel = connection.channel()
    channel.queue_declare(queue="iris", durable=True)
    channel.basic_publish(exchange='', routing_key="iris", body=data_json.encode('utf-8'))
    connection.close()
    reg = requests.get(url + "/predict/iris" )
    print("Request results is: ")
    print(reg.json())

    data = json.loads(
      json.dumps(reg.json()))  # Ensure the JSON is parsed properly

    metrics = data["metrics"]
    print(f"metrics: {metrics}")

    predictions = json.loads(data["predictions"])
    df = pd.DataFrame(predictions)

    # Alkalmazás címe
    st.title("Iris Dataset Predictions")

    # Táblázat megjelenítése
    st.subheader("Prediction Results")
    st.dataframe(df)  # Interaktív táblázat

    # Alap statisztikák
    st.subheader("Statistics")
    st.write(df.describe())

    # Szűrés: Csak adott osztályhoz tartozó predikciók
    selected_class = st.selectbox("Select y_pred class",
                                  options=sorted(df["y_pred"].unique()))
    filtered_df = df[df["y_pred"] == selected_class]
    st.subheader(f"Filtered Results (y_pred = {selected_class})")
    st.dataframe(filtered_df)
    #st.table(data)

   #%% SCores
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.write("Precision")
            st.write(metrics["precision"])

    with col2:
        with st.container(border=True):
            st.write("Accuracy")
            st.write(metrics["accuracy"])

    with col3:
        with st.container(border=True):
            st.write("F1 Score")
            st.write(metrics["f1_score"])

    with col4:
        with st.container(border=True):
            st.write("Recall")
            st.write(metrics["recall"])
    #  #%% figures - heatmap
    # st.pyplot(ConfusionMatrixDisplay.from_predictions(data["Origin"], data["y_pred"]).figure_)

    # AUC
