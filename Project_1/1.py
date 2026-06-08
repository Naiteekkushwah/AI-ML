from flask import Flask,jsonify,request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


data = pd.read_csv("file.csv", encoding="latin-1")
df = pd.DataFrame(data)
adddata = df[["hours_studied", "sleep_hours", "attendance"]]
y=[
    [0], [1],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
     [1],
    [1],
    [0],
    [1],
]

model = KNeighborsClassifier()

model.fit(adddata,y)
@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)
    hours_studied=float(data["study"])
    sleep_hours =float(data["sleep"])
    attendance= float(data["attendance"])

    predect_result = model.predict([[hours_studied,sleep_hours,attendance]])[0]
    print(predect_result,"predect_result")
    return jsonify({
       "result": int(predect_result)
       })




if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )




