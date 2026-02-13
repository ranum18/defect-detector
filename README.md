# Detect Defects

Simple CI/CD demo project for image defect detection.

## Run locally

pip install -r requirements.txt
pytest

## Run with Docker

docker build -t detect-defects .
docker run detect-defects

## CI/CD

GitHub Actions automatically:

- Builds Docker image
- Runs tests

## Author

Anum Rehman
