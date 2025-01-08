FROM quay.io/astronomer/astro-runtime:12.6.0


#FROM apache/airflow:2.6.1
COPY requirements.txt /requirements.txt
COPY ./dags/*.py /usr/local/airflow/dags/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


ENV AIRFLOW_HOME /usr/local/airflow
COPY dags ${AIRFLOW_HOME}/dags