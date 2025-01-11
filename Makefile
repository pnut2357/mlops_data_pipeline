pippoetry:
	pip install --upgrade pip && pip install poetry
install:
	poetry install --with dev
test:
	poetry run python -m pytest -vv --cov=dags --cov-config=.coveragerc
format:
	poetry run black ./dags/*.py
lint:
	poetry run pylint --disable=R,C ./dags/pipeline.py
all: pippoetry install test