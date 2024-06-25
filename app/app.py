from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preds', methods=["POST"])
def my_preds():

    try:
        my_model = load_model('./model_for_iris_problem.keras', compile=False)
    except ValueError as e:
        return f"Error loading model: {str(e)}"

    my_model.compile(
        loss = SparseCategoricalCrossentropy(),
        optimizer = Adam(learning_rate=0.001)
    )

    feature_1 = float(request.form['feature-1-field'])
    feature_2 = float(request.form['feature-2-field'])
    feature_3 = float(request.form['feature-3-field'])
    feature_4 = float(request.form['feature-4-field'])

    acctual_class = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    acctual_class_dict = {i: x for i, x in enumerate(acctual_class)}

    def get_pred_class(x1, x2, x3, x4, model):

        x_np = np.array([[x1, x2, x3, x4]])
        pred_index = np.argmax(model.predict(x_np))
        return acctual_class_dict[pred_index]

    predict_for_input = str(get_pred_class(feature_1, feature_2, feature_3, feature_4, my_model))

    try:
        return render_template('predict.html', predict = predict_for_input)
    except Exception as e:
        return f"{e}"


if __name__ == '__main__':
    app.run(debug=True)