#!/bin/bash

# Wait for the PostgreSQL database to become ready
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Run Airflow database migrations
echo "Running airflow db upgrade..."
airflow db upgrade

# Create an admin user if it doesn't exist
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists."

# Start the Airflow service passed as CMD
exec "$@"