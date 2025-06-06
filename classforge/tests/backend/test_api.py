import pytest
from flask import Flask
import json
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.api.classroom_routes import api as classroom_api

@pytest.fixture
def app():
    """Create a Flask test app with the classroom API"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(classroom_api)
    return app

@pytest.fixture
def client(app):
    """Create a test client for the Flask app"""
    return app.test_client()

class TestClassroomAPI:
    def test_allocate_endpoint_validation(self, client):
        """Test validation in the allocate endpoint"""
        # Test with empty request
        response = client.post('/allocate', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
        assert 'error' in data
        
        # Test with missing students data
        response = client.post('/allocate', json={'params': {}})
        assert response.status_code == 400
        
        # Test with empty students list
        response = client.post('/allocate', json={'students': []})
        assert response.status_code == 400
    
    def test_generate_endpoint(self, client):
        """Test the generate endpoint"""
        response = client.post('/generate', json={'numStudents': 5})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'students' in data
        assert len(data['students']) == 5
