import yaml
from airflow import DAG
from datetime import datetime
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyDatasetOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from astro import sql as aql
from astro.files import File
from airflow.models.baseoperator import chain
from astro.sql.table import Table, Metadata
from astro.constants import FileType
import pandas as pd
import yaml


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def clean_column_names(input_path: str, output_path: str):
    """Cleans column names by removing whitespaces and special characters.
    """
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.replace(r"[^\w]", "", regex=True)
    df.to_csv(output_path, index=False)


with DAG(
    dag_id="stock_pred",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    config = load_config("./include/config.yml")["stock_pred"]
    gcp_conn_id = config["gcp_conn_id"]
    project_id = config["project_id"]
    dataset_id = config["dataset_id"]
    csv_data = config["csv_data"]
    cleaned_csv_data = config["cleaned_csv_data"]
    gcs = config["gcs"]
    dst_path = config["dst_path"]
    saving_table_name = config["saving_table_name"]
    gcs_name = gcs.split('://')[1]
    clean_csv_task = PythonOperator(
        task_id='clean_csv_columns',
        python_callable=clean_column_names,
        op_kwargs={
            'input_path': csv_data,
            'output_path': cleaned_csv_data,
        },
    )
    upload_csv_to_gcs = LocalFilesystemToGCSOperator(
        task_id='upload_csv_to_gcs',
        src=cleaned_csv_data,
        dst=dst_path,
        bucket=gcs_name,
        gcp_conn_id=gcp_conn_id,
        mime_type='text/csv',
    )
    create_dataset = BigQueryCreateEmptyDatasetOperator(
        task_id='create_dataset',
        dataset_id=dataset_id,
        gcp_conn_id=gcp_conn_id,
    )
    gcs_to_raw = GCSToBigQueryOperator(
        task_id="gcs_to_raw",
        bucket=gcs_name,
        source_objects=[dst_path],
        destination_project_dataset_table=f"{project_id}.{dataset_id}.{saving_table_name}",
        source_format="CSV",
        create_disposition="CREATE_IF_NEEDED",
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
        gcp_conn_id=gcp_conn_id,
    )
    chain(clean_csv_task, upload_csv_to_gcs, create_dataset, gcs_to_raw)
