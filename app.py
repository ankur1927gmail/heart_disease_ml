import json
import os
import time

from flask import Flask, Response, g, request, render_template
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest, Histogram

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
application.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

app = application

REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint']
)


@app.before_request
def before_request():
    g.start_time = time.time()


@app.after_request
def after_request(response):
    duration = time.time() - getattr(g, 'start_time', time.time())
    endpoint = request.path
    status = response.status_code
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    logging.info(
        'request=%s path=%s status=%s duration=%.3fs',
        request.method,
        endpoint,
        status,
        duration
    )
    return response


@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


ALLOWED_EXTENSIONS = {'json'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home Page
@app.route('/')
def index():
    return render_template("home.html", active_tab='predict1')


@app.route('/health')
def health():
    return {"status": "ok"}, 200


# Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == "GET":
            return render_template("home.html", active_tab='predict1')

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
        logging.info("Input Data:\n%s", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template(
            "home.html",
            active_tab='predict1',
            results=int(results[0])
        )

    except Exception as e:
        logging.error("Error during single datapoint prediction: %s", str(e))
        return render_template(
            "home.html",
            active_tab='predict1',
            single_error='An error occurred during prediction. Please check the entered values and try again.'
        ), 500


# Bulk prediction route
@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    active_tab = 'bulk_predict'
    file = request.files.get('json_file')

    if not file or file.filename == '':
        return render_template(
            'home.html',
            active_tab=active_tab,
            bulk_error='No file selected. Please choose a JSON file.'
        )

    if not allowed_file(file.filename):
        return render_template(
            'home.html',
            active_tab=active_tab,
            bulk_error='Invalid file type. Only JSON files are allowed.',
            bulk_filename=file.filename
        )

    try:
        logging.info("Bulk prediction upload received: %s", file.filename)
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > app.config['MAX_CONTENT_LENGTH']:
            raise ValueError('File size exceeds maximum limit of 200MB.')

        file_content = file.read().decode('utf-8')
        data = json.loads(file_content)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        required_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal'
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return render_template(
                'home.html',
                active_tab=active_tab,
                bulk_error=f'Missing required columns: {", ".join(missing)}',
                bulk_filename=file.filename
            )

        df = df[required_columns].copy()
        logging.info("Bulk predict DataFrame shape: %s", df.shape)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df)

        df['predicted'] = [int(x) for x in results]

        return render_template(
            'home.html',
            active_tab=active_tab,
            bulk_filename=file.filename,
            bulk_table=df.to_dict(orient='records'),
            bulk_columns=list(df.columns)
        )

    except json.JSONDecodeError:
        return render_template(
            'home.html',
            active_tab=active_tab,
            bulk_error='Invalid JSON format. Please upload a valid JSON file.',
            bulk_filename=file.filename
        )
    except ValueError as ve:
        return render_template(
            'home.html',
            active_tab=active_tab,
            bulk_error=str(ve),
            bulk_filename=file.filename
        )
    except Exception as e:
        logging.error("Unhandled exception during bulk prediction: %s", str(e))
        return render_template(
            'home.html',
            active_tab=active_tab,
            bulk_error=f'An error occurred during bulk prediction: {str(e)}',
            bulk_filename=file.filename
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True,port=8000)