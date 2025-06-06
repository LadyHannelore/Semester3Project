"""
Tests for the database models and operations
"""
import pytest
from datetime import datetime
import json
from backend.models.database import Student, ClassAssignment, AllocationRun
from backend.models.migrations import init_db


def test_student_model(app):
    """Test Student model creation and conversion"""
    with app.app_context():
        # Create a student
        student = Student(
            id="1001",
            academic_score=85.2,
            wellbeing_score=3.5,
            bullying_score=2.1,
            friends_json="1002,1003,1004"
        )
        
        # Test properties
        assert student.id == "1001"
        assert student.academic_score == 85.2
        assert student.wellbeing_score == 3.5
        assert student.bullying_score == 2.1
        assert student.friends_json == "1002,1003,1004"
        assert student.friends == ["1002", "1003", "1004"]
        
        # Test to_dict method
        student_dict = student.to_dict()
        assert student_dict["id"] == "1001"
        assert student_dict["academicScore"] == 85.2
        assert student_dict["wellbeingScore"] == 3.5
        assert student_dict["bullyingScore"] == 2.1
        assert student_dict["friends"] == "1002,1003,1004"
        
        # Test from_dict method
        input_dict = {
            "id": "2001",
            "academicScore": 90.5,
            "wellbeingScore": 4.2,
            "bullyingScore": 1.5,
            "friends": "2002,2003"
        }
        new_student = Student.from_dict(input_dict)
        assert new_student.id == "2001"
        assert new_student.academic_score == 90.5
        assert new_student.wellbeing_score == 4.2
        assert new_student.bullying_score == 1.5
        assert new_student.friends_json == "2002,2003"


def test_allocation_run_model(app, db):
    """Test AllocationRun model"""
    with app.app_context():
        # Create an allocation run
        params = {
            "maxClassSize": 25,
            "maxBulliesPerClass": 2,
            "generations": 50
        }
        metrics = {
            "averageScore": 85.5,
            "classBalance": 0.92
        }
        violations = {
            "sizeViolations": 0,
            "bullyingViolations": 1
        }
        
        allocation_run = AllocationRun(
            algorithm_type="genetic_algorithm",
            parameters_json=json.dumps(params),
            metrics_json=json.dumps(metrics),
            violations_json=json.dumps(violations)
        )
        
        # Add to database
        db.session.add(allocation_run)
        db.session.commit()
        
        # Retrieve from database
        retrieved_run = AllocationRun.query.get(allocation_run.id)
        
        # Verify data
        assert retrieved_run.algorithm_type == "genetic_algorithm"
        assert json.loads(retrieved_run.parameters_json) == params
        assert json.loads(retrieved_run.metrics_json) == metrics
        assert json.loads(retrieved_run.violations_json) == violations
        assert isinstance(retrieved_run.created_at, datetime)


def test_class_assignment_model(app, db):
    """Test ClassAssignment model with relationships"""
    with app.app_context():
        # Create a student
        student = Student(
            id="3001",
            academic_score=82.1,
            wellbeing_score=3.8,
            bullying_score=1.9,
            friends_json="3002,3003"
        )
        db.session.add(student)
        
        # Create an allocation run
        allocation_run = AllocationRun(
            algorithm_type="genetic_algorithm",
            parameters_json=json.dumps({"maxClassSize": 25})
        )
        db.session.add(allocation_run)
        db.session.commit()
        
        # Create class assignment
        assignment = ClassAssignment(
            student_id=student.id,
            class_id=1,
            allocation_run_id=allocation_run.id
        )
        db.session.add(assignment)
        db.session.commit()
        
        # Test relationships
        assert assignment.student.id == "3001"
        assert assignment.allocation_run.id == allocation_run.id
        assert assignment in student.assignments
        assert assignment in allocation_run.assignments
        
        # Test to_dict method
        assignment_dict = assignment.to_dict()
        assert assignment_dict["studentId"] == "3001"
        assert assignment_dict["classId"] == 1
        assert assignment_dict["allocationRunId"] == allocation_run.id


def test_database_integration(app, db):
    """Test database integration with all models"""
    with app.app_context():
        # Create multiple students
        students = [
            Student(id="4001", academic_score=85.0, wellbeing_score=3.5, bullying_score=2.0, friends_json="4002,4003"),
            Student(id="4002", academic_score=90.0, wellbeing_score=4.0, bullying_score=1.0, friends_json="4001"),
            Student(id="4003", academic_score=75.0, wellbeing_score=3.0, bullying_score=2.5, friends_json="4001,4002")
        ]
        db.session.add_all(students)
        
        # Create allocation run
        allocation_run = AllocationRun(
            algorithm_type="genetic_algorithm",
            parameters_json=json.dumps({"maxClassSize": 25})
        )
        db.session.add(allocation_run)
        db.session.commit()
        
        # Create assignments
        assignments = [
            ClassAssignment(student_id="4001", class_id=1, allocation_run_id=allocation_run.id),
            ClassAssignment(student_id="4002", class_id=1, allocation_run_id=allocation_run.id),
            ClassAssignment(student_id="4003", class_id=2, allocation_run_id=allocation_run.id)
        ]
        db.session.add_all(assignments)
        db.session.commit()
        
        # Verify class distribution
        class1_students = ClassAssignment.query.filter_by(class_id=1, allocation_run_id=allocation_run.id).all()
        class2_students = ClassAssignment.query.filter_by(class_id=2, allocation_run_id=allocation_run.id).all()
        
        assert len(class1_students) == 2
        assert len(class2_students) == 1
        assert class1_students[0].student_id == "4001"
        assert class1_students[1].student_id == "4002"
        assert class2_students[0].student_id == "4003"
