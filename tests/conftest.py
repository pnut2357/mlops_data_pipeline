import pytest
from airflow.models import DagBag
import pandas as pd
import uuid


@pytest.fixture(scope="session")
def dag_bag():
    """Fixture to load the DagBag once for the test session.
    """
    return DagBag(include_examples=False)

@pytest.fixture
def sample_csv_file(tmp_path):
    """Creates a sample CSV file for testing.
    """
    data = {
        "Turnover (Lacs)": [1234, 5678],
        "Profit %": [20.5, 15.3],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / f"sample_{uuid.uuid4()}.csv"
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def clean_csv_file(tmp_path):
    """Creates a cleaned CSV file for testing.
    """
    data = {
        "Turnover_Lacs": [1234, 5678],
        "Profit_": [20.5, 15.3],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / f"cleaned_{uuid.uuid4()}.csv"
    df.to_csv(file_path, index=False)
    return file_path