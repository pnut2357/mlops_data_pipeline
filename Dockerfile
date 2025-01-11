FROM quay.io/astronomer/astro-runtime:12.6.0
USER root

RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN airflow db init
ENV AIRFLOW_HOME=/usr/local/airflow
#ENV AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8181
ENV AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.session
COPY ./dags ${AIRFLOW_HOME}/dags
COPY requirements.txt /tmp/requirements.txt
RUN chmod +x /usr/local/airflow/dags/*

USER astro
RUN python -m pip install --upgrade pip
RUN pip install poetry
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN airflow version

#CMD ["astro", "dev", "start"]
