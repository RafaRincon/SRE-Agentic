import pytest
from fastapi.testclient import TestClient

# Adding app path to simulate what we need if running tests
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

@pytest.fixture
def client():
    """Returns a test client for the FastAPI application."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_incident_report():
    return "HTTP 500 in Checkout. NullReferenceException at OrdersController.cs:42"

@pytest.fixture
def sample_hypothesis():
    return {
        "title": "NullReference in Order Processing",
        "description": "The order object is null when processing checkout.",
        "citations": [
            {
                "file_name": "OrdersController.cs",
                "start_line": 40,
                "end_line": 45
            }
        ]
    }
