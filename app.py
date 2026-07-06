from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Home Page
@app.route('/')
def index():
    return render_template("home.html")


# Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == "GET":
        return render_template("home.html")

    else:

        data = CustomData(

            age=float(request.form.get("age")),
            sex=int(request.form.get("sex")),
            cp=int(request.form.get("cp")),
            trestbps=float(request.form.get("trestbps")),
            chol=float(request.form.get("chol")),
            fbs=int(request.form.get("fbs")),
            restecg=int(request.form.get("restecg")),
            thalach=float(request.form.get("thalach")),
            exang=int(request.form.get("exang")),
            oldpeak=float(request.form.get("oldpeak")),
            slope=int(request.form.get("slope")),
            ca=int(request.form.get("ca")),
            thal=int(request.form.get("thal"))

        )

        pred_df = data.get_data_as_data_frame()

        print("Input Data")
        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template(
            "home.html",
            results=int(results[0])
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)