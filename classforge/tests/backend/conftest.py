# Test configuration for ClassForge backend
import os
import sys
import pytest
import tempfile

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.main import create_app
from backend.models.database import db as _db


@pytest.fixture
def app():
    """Create and configure a Flask app for testing"""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()
    
    # Create the app with test config
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'WTF_CSRF_ENABLED': False  # Disable CSRF protection in tests
    })
    
    # Create the database and tables
    with app.app_context():
        _db.create_all()
    
    yield app
    
    # Clean up
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """Test client for our Flask app"""
    return app.test_client()


@pytest.fixture
def db(app):
    """Database for testing"""
    with app.app_context():
        _db.create_all()
        
    yield _db
    
    # Clean up
    with app.app_context():
        _db.session.remove()
        _db.drop_all()
