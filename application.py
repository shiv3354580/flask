from flask import Flask, render_template, request, redirect, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

import os
def generate_price_range(predicted_price) -> dict:

    if predicted_price < 1000000:
        result = predicted_price - predicted_price * 0.15
    else:
        result = predicted_price - predicted_price * 0.1
    return {"min_price": round(result, -4), "max_price": predicted_price}


# result = generate_price_range(48500)

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("Cleaned_Car_data.csv")


@app.route("/predict-car-price", methods=["POST"])
def add_income():
    data = request.get_json()
    company = data.get("company")
    car_model = data.get("car_models")
    year = data.get("year")
    fuel_type = data.get("fuel_type")
    driven = data.get("kilo_driven")
    print(data)
    prediction = model.predict(
        pd.DataFrame(
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5),
        )
    )
    print(prediction)
    response = jsonify(generate_price_range(np.round(prediction[0], 2)))
    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


@app.route("/", methods=["GET", "POST"])
def index():
    print(pd.read_pickle("LinearRegressionModel.pkl"), "reading picle")

    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()

    companies.insert(0, "Select Company")
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_type,
    )


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    print("Predicting")
    company = request.form.get("company")

    car_model = request.form.get("car_models")
    year = request.form.get("year")
    fuel_type = request.form.get("fuel_type")
    driven = request.form.get("kilo_driven")
    print(company, car_model, year, fuel_type, driven)
    print(model)
    prediction = model.predict(
        pd.DataFrame(
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5),
        )
    )
    # print(prediction)

    return str(np.round(prediction[0], 2)) # return json
    # return str(1)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))