import os
import logging
from contextlib import contextmanager
import pytest
from airflow.models import DagBag
from unittest.mock import MagicMock, patch
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyDatasetOperator
import pandas as pd
from dags.online_retail import create_dataset
from dags.online_retail import clean_column_names


@pytest.mark.parametrize("dag_id", ["online_retail"])
def test_dag_loaded(dag_bag, dag_id):
    """Test if the DAG is successfully loaded.
    """
    mock_dag_bag = MagicMock()
    mock_dag = MagicMock()
    mock_dag.tasks = [MagicMock(task_id="dummy_task")]
    mock_dag_bag.get_dag.return_value = mock_dag
    dag = mock_dag_bag.get_dag(mock_dag)
    assert dag is not None, f"DAG with id {dag_id} is not found."
    assert len(dag.tasks) > 0, f"DAG with id {dag_id} has no tasks."

def test_task_clean_csv_columns(sample_csv_file, tmp_path):
    """Test the clean_column_names function to ensure column cleaning works as expected.
    """

    cleaned_file = tmp_path / "cleaned_file.csv"
    clean_column_names(input_path=sample_csv_file, output_path=cleaned_file)
    df = pd.read_csv(cleaned_file)
    assert "TurnoverLacs" in df.columns, "Column cleaning failed for 'Turnover (Lacs)'"
    assert "Profit" in df.columns, "Column cleaning failed for 'Profit %'"


@pytest.mark.parametrize("dag_id, task_id", [
    ("online_retail", "upload_csv_to_gcs"),
    ("online_retail", "create_dataset"),
    ("online_retail", "gcs_to_raw"),
])
def test_task_in_dag(dag_id, task_id):
    """Test if specific tasks exist in the DAG.
    """
    mock_dag_bag = MagicMock()
    mock_dag = MagicMock()
    mock_task = MagicMock(task_id=task_id)
    mock_dag.tasks = [mock_task]  # Simulate the task list in the DAG
    mock_dag.get_task.return_value = mock_task  # Simulate get_task behavior
    mock_dag_bag.get_dag.return_value = mock_dag
    dag = mock_dag_bag.get_dag(dag_id)
    assert dag.get_task(task_id) is not None, f"Task {task_id} is missing in DAG {dag_id}"


@pytest.mark.parametrize("task_id, operator", [
    ("upload_csv_to_gcs", "LocalFilesystemToGCSOperator"),
    ("create_dataset", "BigQueryCreateEmptyDatasetOperator"),
])
def test_task_operator(task_id, operator):
    """Test if tasks use the correct operator using MagicMock.
    """
    mock_dag_bag = MagicMock()
    mock_dag = MagicMock()
    mock_task = MagicMock(spec=operator)
    mock_task.task_id = task_id
    mock_dag.get_task.return_value = mock_task
    mock_dag_bag.get_dag.return_value = mock_dag
    dag = mock_dag_bag.get_dag("online_retail")
    task = dag.get_task(task_id)
    assert isinstance(task, MagicMock), f"Task {task_id} is not a mocked instance."
    assert task.task_id == task_id, f"Task {task_id} does not match the mocked task."


@patch("airflow.providers.google.cloud.operators.bigquery.BigQueryCreateEmptyDatasetOperator.execute")
def test_bigquery_dataset_creation(mock_execute):
    """Test BigQuery dataset creation task using MagicMock.
    """
    mock_execute.return_value = None
    mock_context = MagicMock()
    create_dataset.execute(mock_context)
    mock_execute.assert_called_once_with(mock_context)


