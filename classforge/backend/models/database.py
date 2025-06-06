"""
Database configuration and models for ClassForge backend.
This module sets up SQLAlchemy for database access.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

# Create SQLAlchemy instance
db = SQLAlchemy()

class Student(db.Model):
    """Student model for database storage"""
    __tablename__ = 'students'
    
    id = db.Column(db.String(20), primary_key=True)
    academic_score = db.Column(db.Float, nullable=False)
    wellbeing_score = db.Column(db.Float, nullable=False)
    bullying_score = db.Column(db.Float, nullable=False)
    friends_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define relationship with ClassAssignment
    assignments = db.relationship('ClassAssignment', back_populates='student', cascade='all, delete-orphan')
    
    @property
    def friends(self):
        """Get list of friend IDs from JSON string"""
        if not self.friends_json:
            return []
        return self.friends_json.split(',')
    
    @friends.setter
    def friends(self, friend_ids):
        """Set list of friend IDs as JSON string"""
        if isinstance(friend_ids, list):
            self.friends_json = ','.join(friend_ids)
        else:
            self.friends_json = friend_ids
    
    def to_dict(self):
        """Convert student to dictionary representation"""
        return {
            'id': self.id,
            'academicScore': self.academic_score,
            'wellbeingScore': self.wellbeing_score,
            'bullyingScore': self.bullying_score,
            'friends': self.friends_json or '',
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create student from dictionary"""
        return cls(
            id=str(data.get('id')),
            academic_score=float(data.get('academicScore', data.get('Academic_Performance', 0))),
            wellbeing_score=float(data.get('wellbeingScore', data.get('Wellbeing_Score', 0))),
            bullying_score=float(data.get('bullyingScore', data.get('Bullying_Score', 0))),
            friends_json=data.get('friends', data.get('Friends', ''))
        )


class ClassAssignment(db.Model):
    """Class assignment model for database storage"""
    __tablename__ = 'class_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('students.id'), nullable=False)
    class_id = db.Column(db.Integer, nullable=False)
    allocation_run_id = db.Column(db.Integer, db.ForeignKey('allocation_runs.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define relationships
    student = db.relationship('Student', back_populates='assignments')
    allocation_run = db.relationship('AllocationRun', back_populates='assignments')
    
    def to_dict(self):
        """Convert class assignment to dictionary representation"""
        return {
            'id': self.id,
            'studentId': self.student_id,
            'classId': self.class_id,
            'allocationRunId': self.allocation_run_id,
            'createdAt': self.created_at.isoformat() if self.created_at else None
        }


class AllocationRun(db.Model):
    """Allocation run model for database storage"""
    __tablename__ = 'allocation_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    algorithm_type = db.Column(db.String(50), nullable=False)
    parameters_json = db.Column(db.Text, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)
    violations_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(100), nullable=True)
    
    # Define relationship with ClassAssignment
    assignments = db.relationship('ClassAssignment', back_populates='allocation_run', cascade='all, delete-orphan')
    
    @property
    def parameters(self):
        """Get parameters from JSON string"""
        if not self.parameters_json:
            return {}
        return json.loads(self.parameters_json)
    
    @parameters.setter
    def parameters(self, params):
        """Set parameters as JSON string"""
        if isinstance(params, dict):
            self.parameters_json = json.dumps(params)
        else:
            self.parameters_json = params
    
    @property
    def metrics(self):
        """Get metrics from JSON string"""
        if not self.metrics_json:
            return {}
        return json.loads(self.metrics_json)
    
    @metrics.setter
    def metrics(self, metrics_dict):
        """Set metrics as JSON string"""
        if isinstance(metrics_dict, dict):
            self.metrics_json = json.dumps(metrics_dict)
        else:
            self.metrics_json = metrics_dict
    
    @property
    def violations(self):
        """Get violations from JSON string"""
        if not self.violations_json:
            return []
        return json.loads(self.violations_json)
    
    @violations.setter
    def violations(self, violations_list):
        """Set violations as JSON string"""
        if isinstance(violations_list, list):
            self.violations_json = json.dumps(violations_list)
        else:
            self.violations_json = violations_list
    
    def to_dict(self):
        """Convert allocation run to dictionary representation"""
        return {
            'id': self.id,
            'algorithmType': self.algorithm_type,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'violations': self.violations,
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'createdBy': self.created_by
        }
