import io
import json
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# These imports are intentionally placed after the path setup so the package
# can be imported reliably during test execution.
from app import app, allowed_file  # noqa: E402
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # noqa: E402


# Fixture to change the working directory into a temporary path for each test.
# This keeps artifact files from being written to the real repository.
@pytest.fixture(autouse=True)
def set_temp_artifacts_dir(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return artifacts_dir


# Fixture that builds a small dummy preprocessor and model on disk.
# The PredictPipeline reads these files from artifacts/model.pkl and
# artifacts/preprocessor.pkl.
@pytest.fixture()
def build_dummy_preprocessor_and_model(tmp_path):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import pickle

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = StandardScaler()
    train = [
        [45.0, 0.0, 2.0, 130.0, 250.0, 0.0, 1.0, 150.0, 0.0, 1.0,
         2.0, 0.0, 3.0],
        [50.0, 1.0, 3.0, 140.0, 240.0, 1.0, 0.0, 160.0, 1.0, 0.8,
         2.0, 0.0, 6.0],
        [55.0, 0.0, 1.0, 135.0, 260.0, 0.0, 1.0, 145.0, 0.0, 1.2,
         1.0, 1.0, 7.0],
    ]
    preprocessor.fit(train)
    with open(artifacts_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    model = LogisticRegression(max_iter=1000)
    X = train
    y = [0, 1, 1]
    model.fit(X, y)
    with open(artifacts_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    return artifacts_dir


def test_allowed_file():
    """Verify the file type check only allows JSON files."""
    assert allowed_file("data.json")
    assert not allowed_file("data.csv")
    assert not allowed_file("noextension")


def test_custom_data_to_dataframe():
    """Ensure CustomData correctly builds a one-row DataFrame."""
    data = CustomData(
        age=55,
        sex=1,
        cp=3,
        trestbps=140,
        chol=220,
        fbs=0,
        restecg=1,
        thalach=150,
        exang=0,
        oldpeak=1.2,
        slope=2,
        ca=0,
        thal=3,
    )
    df = data.get_data_as_data_frame()

    assert list(df.columns) == [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal"
    ]
    assert df.loc[0, "age"] == 55
    assert df.loc[0, "thal"] == 3


def test_predict_pipeline_with_dummy_model(build_dummy_preprocessor_and_model):
    """Validate PredictPipeline can load artifacts and produce a prediction."""
    df = pd.DataFrame({
        "age": [45.0],
        "sex": [0.0],
        "cp": [2.0],
        "trestbps": [130.0],
        "chol": [250.0],
        "fbs": [0.0],
        "restecg": [1.0],
        "thalach": [150.0],
        "exang": [0.0],
        "oldpeak": [1.0],
        "slope": [2.0],
        "ca": [0.0],
        "thal": [3.0],
    })

    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    # The result should be a one-element prediction array from the dummy model.
    assert len(result) == 1
    assert int(result[0]) in {0, 1}


def test_bulk_predict_success(monkeypatch, build_dummy_preprocessor_and_model):
    """Test the bulk-predict endpoint with a valid JSON upload."""
    client = app.test_client()
    payload = [
        {
            "age": 50,
            "sex": 1,
            "cp": 3,
            "trestbps": 140,
            "chol": 230,
            "fbs": 0,
            "restecg": 1,
            "thalach": 155,
            "exang": 0,
            "oldpeak": 0.8,
            "slope": 2,
            "ca": 0,
            "thal": 3,
        }
    ]

    data = json.dumps(payload).encode("utf-8")
    response = client.post(
        "/bulk_predict",
        data={"json_file": (io.BytesIO(data), "bulk.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Showing predictions for" in response.data
    assert b"bulk.json" in response.data


def test_bulk_predict_missing_columns(
    monkeypatch,
    build_dummy_preprocessor_and_model,
):
    """Ensure bulk prediction returns an error when required fields are
    missing."""
    client = app.test_client()
    payload = [{"age": 50, "sex": 1}]  # missing many required columns
    data = json.dumps(payload).encode("utf-8")

    response = client.post(
        "/bulk_predict",
        data={"json_file": (io.BytesIO(data), "bulk.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Missing required columns" in response.data


def test_bulk_predict_invalid_json(monkeypatch):
    """Check JSON decode errors are caught and reported gracefully."""
    client = app.test_client()
    response = client.post(
        "/bulk_predict",
        data={"json_file": (io.BytesIO(b"not-json"), "bulk.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Invalid JSON format" in response.data


def test_bulk_predict_invalid_file_type(monkeypatch):
    """Verify the upload rejects non-JSON file extensions."""
    client = app.test_client()
    response = client.post(
        "/bulk_predict",
        data={"json_file": (io.BytesIO(b"[]"), "bulk.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Invalid file type" in response.data


def test_predict_datapoint_exception(monkeypatch):
    """Check the single prediction route wraps errors using CustomException."""
    client = app.test_client()

    # Force an exception by sending a missing required field.
    response = client.post(
        "/predictdata",
        data={
            "age": "50",
            "sex": "1",
            # Missing most required fields intentionally.
        },
        content_type="application/x-www-form-urlencoded",
    )

    assert response.status_code == 500
    assert (
        b"An error occurred" in response.data
        or b"Error occured" in response.data
    )
