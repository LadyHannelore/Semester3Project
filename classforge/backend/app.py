# Main Flask application for ClassForge
from flask import Flask, request, jsonify
from flask_cors import CORS
from algorithms.genetic_algorithm import solve_with_genetic_algorithm
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the frontend

@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing API documentation."""
    return jsonify({
        'service': 'ClassForge API',
        'version': '1.0',
        'endpoints': {
            '/allocate': 'POST - Submit students data and parameters to generate classroom allocations',
            '/health': 'GET - Check API health status'
        },
        'status': 'running'
    })

@app.route('/allocate', methods=['POST'])
def allocate():
    """Process classroom allocation request"""
    try:
        data = request.get_json()
        students_data = data['students']
        params = data.get('params', {})
        
        # Convert to DataFrame
        df = pd.DataFrame(students_data)
        
        # Run allocation algorithm
        result = solve_with_genetic_algorithm(
            df,
            class_size_limit=params.get('maxClassSize', 25),
            max_bullies_per_class=params.get('maxBulliesPerClass', 2),
            wellbeing_min=params.get('wellbeingMin'),
            wellbeing_max=params.get('wellbeingMax'),
            generations=params.get('generations', 50),
            pop_size=params.get('populationSize', 100)
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(port=5001, debug=False)
