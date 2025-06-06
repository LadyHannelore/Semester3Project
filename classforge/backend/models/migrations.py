"""
Database initialization and migration utilities for ClassForge.
This module provides functions to initialize and migrate the database.
"""

import os
import click
from flask import Flask
from flask.cli import with_appcontext
from backend.models.database import db
from alembic import command
from alembic.config import Config
import logging

logger = logging.getLogger(__name__)

def init_db(app):
    """Initialize the database with the application context"""
    # Create all tables
    with app.app_context():
        logger.info("Initializing database tables...")
        db.create_all()
        logger.info("Database tables created.")

def get_alembic_config(app):
    """Get Alembic configuration"""
    # Get directory where this file is located
    directory = os.path.dirname(os.path.abspath(__file__))
    
    # Create Alembic config
    config = Config(os.path.join(directory, "../migrations/alembic.ini"))
    config.set_main_option("script_location", os.path.join(directory, "../migrations"))
    config.set_main_option("sqlalchemy.url", app.config["SQLALCHEMY_DATABASE_URI"])
    
    return config

def init_migrations(app):
    """Initialize database migrations"""
    # Get Alembic config
    config = get_alembic_config(app)
    
    # Create migrations directory if it doesn't exist
    migrations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../migrations")
    if not os.path.exists(migrations_dir):
        os.makedirs(migrations_dir)
    
    # Initialize migrations
    logger.info("Initializing database migrations...")
    command.init(config, migrations_dir)
    logger.info("Database migrations initialized.")

def migrate_db(app):
    """Generate a new migration based on changes to models"""
    # Get Alembic config
    config = get_alembic_config(app)
    
    # Generate migration
    logger.info("Generating database migration...")
    command.revision(config, autogenerate=True, message="Auto-generated migration")
    logger.info("Database migration generated.")

def upgrade_db(app):
    """Upgrade database to latest revision"""
    # Get Alembic config
    config = get_alembic_config(app)
    
    # Upgrade database
    logger.info("Upgrading database to latest revision...")
    command.upgrade(config, "head")
    logger.info("Database upgraded.")

@click.command("init-db")
@with_appcontext
def init_db_command():
    """Initialize the database."""
    from flask import current_app
    init_db(current_app)
    click.echo("Database initialized.")

@click.command("init-migrations")
@with_appcontext
def init_migrations_command():
    """Initialize database migrations."""
    from flask import current_app
    init_migrations(current_app)
    click.echo("Database migrations initialized.")

@click.command("migrate")
@with_appcontext
def migrate_db_command():
    """Generate a new migration."""
    from flask import current_app
    migrate_db(current_app)
    click.echo("Database migration generated.")

@click.command("upgrade")
@with_appcontext
def upgrade_db_command():
    """Upgrade the database."""
    from flask import current_app
    upgrade_db(current_app)
    click.echo("Database upgraded.")
    
def register_db_cli(app):
    """Register database CLI commands with the Flask application."""
    app.cli.add_command(init_db_command)
    app.cli.add_command(init_migrations_command)
    app.cli.add_command(migrate_db_command)
    app.cli.add_command(upgrade_db_command)
    click.echo("Database upgraded.")

def register_db_cli(app):
    """Register database CLI commands with the Flask application"""
    app.cli.add_command(init_db_command)
    app.cli.add_command(init_migrations_command)
    app.cli.add_command(migrate_db_command)
    app.cli.add_command(upgrade_db_command)
