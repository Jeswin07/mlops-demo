import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import json
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-demo")

with mlflow.start_run():

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names) #type:ignore
    df["target"] = data.target #type:ignore
    df.to_csv("data/dataset.csv", index=False)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    n_estimators = 100
    model = RandomForestClassifier(n_estimators= n_estimators)
    model.fit(X_train,y_train)

    joblib.dump(model, "models/model.pkl")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)

    print("accuracy:",acc)

    with open("metrics.json", "w") as f:
        json.dump({"accuracy":acc},f)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)#type:ignore
    mlflow.sklearn.log_model(model,name="model") #type:ignore

    if acc < 0.8:
        open("retain.flag", "w").close()

    print("Pipeline complete")
