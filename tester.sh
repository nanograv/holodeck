# python convert_notebook_tests.py
pytest -v --cov=holodeck --cov-report=xml --color=yes holodeck/tests/
