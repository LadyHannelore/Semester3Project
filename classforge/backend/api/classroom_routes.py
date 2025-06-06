# API routes for classroom allocation
from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.algorithms.genetic_algorithm import solve_with_genetic_algorithm
from backend.utils.data_processing import generate_synthetic_data
from backend.models.database import db, Student, ClassAssignment, AllocationRun

# Create blueprint
api = Blueprint('api', __name__)

@api.route('/allocate', methods=['POST'])
def allocate_classrooms():
    """
    Endpoint to allocate students to classrooms using genetic algorithm
    
    Expected JSON payload:
    {
        'students': [
            {
                'id': '1001',
                'academicScore': 85.2,
                'wellbeingScore': 3.5,
                'bullyingScore': 2.1,
                'friends': '1005,1008,1012'
            },
            ...
        ],
        'params': {
            'maxClassSize': 25,
            'maxBulliesPerClass': 2,
            'wellbeingMin': 3.0,
            'wellbeingMax': null,
            'generations': 50,
            'populationSize': 100
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate request data
        if not data or 'students' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required student data'
            }), 400
            
        students_data = data['students']
        params = data.get('params', {})
        
        # Filter out students with missing 'id'
        valid_students_data = [s for s in students_data if s.get('id') is not None]
        if not valid_students_data:
            return jsonify({
                'success': False,
                'error': 'No valid student data received (all students missing ID).'
            }), 400
        
        # Store students in database
        students_dict = {}
        for student_data in valid_students_data:
            student_id = str(student_data.get('id'))
            # Check if student already exists
            student = Student.query.get(student_id)
            
            if not student:
                # Create new student
                student = Student.from_dict(student_data)
                db.session.add(student)
            else:
                # Update existing student
                student.academic_score = float(student_data.get('academicScore', student_data.get('Academic_Performance', 0)))
                student.wellbeing_score = float(student_data.get('wellbeingScore', student_data.get('Wellbeing_Score', 0)))
                student.bullying_score = float(student_data.get('bullyingScore', student_data.get('Bullying_Score', 0)))
                student.friends_json = student_data.get('friends', student_data.get('Friends', ''))
            
            students_dict[student_id] = student
                
        # Commit students to database
        db.session.commit()
        
        # Map JS fields to DataFrame columns
        df = pd.DataFrame([{
            'StudentID': str(s.get('id')),
            'Academic_Performance': s.get('academicScore', s.get('Academic_Performance')),
            'Wellbeing_Score': s.get('wellbeingScore', s.get('Wellbeing_Score')),
            'Bullying_Score': s.get('bullyingScore', s.get('Bullying_Score')),
            'Friends': s.get('friends', s.get('Friends'))
        } for s in valid_students_data])
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'DataFrame is empty after processing student data.'
            }), 400
            
        # Run GA
        result = solve_with_genetic_algorithm(
            df,
            class_size_limit=params.get('maxClassSize', 25),
            max_bullies_per_class=params.get('maxBulliesPerClass', 2),
            wellbeing_min=params.get('wellbeingMin'),
            wellbeing_max=params.get('wellbeingMax'),
            generations=params.get('generations', 50),
            pop_size=params.get('populationSize', 100)
        )
        
        # Create allocation run in database
        allocation_run = AllocationRun(
            algorithm_type='genetic_algorithm',
            parameters_json=json.dumps(params),
            metrics_json=json.dumps(result.get('metrics', {})),
            violations_json=json.dumps(result.get('violations', {}))
        )
        db.session.add(allocation_run)
        db.session.commit()
        
        # Create class assignments in database
        assignments = result.get('classes', {})
        for class_id, student_ids in assignments.items():
            for student_id in student_ids:
                assignment = ClassAssignment(
                    student_id=str(student_id),
                    class_id=int(class_id),
                    allocation_run_id=allocation_run.id
                )
                db.session.add(assignment)
        
        # Commit assignments to database
        db.session.commit()
        
        # Include run ID in result
        result['allocationRunId'] = allocation_run.id
        
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Error allocating classrooms: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/generate', methods=['POST'])
def generate_data():
    """
    Generate synthetic student data for testing
    
    Expected JSON payload:
    {
        'numStudents': 1000
    }
    """
    try:
        data = request.get_json()
        num_students = data.get('numStudents', 1000)
        
        # Generate data
        df = generate_synthetic_data(num_students)
        
        # Convert to list of dictionaries for JSON response
        students = []
        for _, row in df.iterrows():
            students.append({
                'id': row['StudentID'],
                'academicScore': row['Academic_Performance'],
                'wellbeingScore': row['Wellbeing_Score'],
                'bullyingScore': row['Bullying_Score'],
                'friends': row['Friends']
            })
            
        return jsonify({
            'success': True,
            'students': students
        })
    except Exception as e:
        current_app.logger.error(f"Error generating data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/allocations', methods=['GET'])
def get_allocations():
    """Get list of all allocation runs"""
    try:
        runs = AllocationRun.query.order_by(AllocationRun.created_at.desc()).all()
        
        result = []
        for run in runs:
            result.append({
                'id': run.id,
                'algorithmType': run.algorithm_type,
                'parameters': json.loads(run.parameters_json) if run.parameters_json else {},
                'metrics': json.loads(run.metrics_json) if run.metrics_json else {},
                'violations': json.loads(run.violations_json) if run.violations_json else {},
                'createdAt': run.created_at.isoformat()
            })
            
        return jsonify({
            'success': True,
            'allocations': result
        })
    except Exception as e:
        current_app.logger.error(f"Error retrieving allocations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/allocations/<int:run_id>', methods=['GET'])
def get_allocation(run_id):
    """Get details of a specific allocation run"""
    try:
        run = AllocationRun.query.get(run_id)
        
        if not run:
            return jsonify({
                'success': False,
                'error': f'Allocation run {run_id} not found'
            }), 404
            
        # Get assignments for this run
        assignments = ClassAssignment.query.filter_by(allocation_run_id=run_id).all()
        
        # Group by class
        classes = {}
        for assignment in assignments:
            class_id = assignment.class_id
            if class_id not in classes:
                classes[class_id] = []
            classes[class_id].append(assignment.student_id)
            
        result = {
            'id': run.id,
            'algorithmType': run.algorithm_type,
            'parameters': json.loads(run.parameters_json) if run.parameters_json else {},
            'metrics': json.loads(run.metrics_json) if run.metrics_json else {},
            'violations': json.loads(run.violations_json) if run.violations_json else {},
            'classes': classes,
            'createdAt': run.created_at.isoformat()
        }
            
        return jsonify({
            'success': True,
            'allocation': result
        })
    except Exception as e:
        current_app.logger.error(f"Error retrieving allocation {run_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/students', methods=['GET'])
def get_students():
    """Get list of all students in the database"""
    try:
        students = Student.query.all()
        
        result = [student.to_dict() for student in students]
            
        return jsonify({
            'success': True,
            'students': result
        })
    except Exception as e:
        current_app.logger.error(f"Error retrieving students: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
