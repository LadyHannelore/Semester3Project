import pytest
import pandas as pd
import numpy as np
from backend.algorithms.genetic_algorithm import solve_with_genetic_algorithm
from backend.utils.data_processing import generate_synthetic_data

class TestGeneticAlgorithm:
    @pytest.fixture
    def sample_data(self):
        """Generate a small sample dataset for testing"""
        # Create a small synthetic dataset
        df = pd.DataFrame({
            'StudentID': ['1001', '1002', '1003', '1004', '1005'],
            'Academic_Performance': [70, 80, 60, 90, 75],
            'Wellbeing_Score': [3.5, 4.0, 2.5, 3.0, 3.8],
            'Bullying_Score': [2.0, 1.0, 8.0, 3.0, 1.5],
            'Friends': ['1002,1004', '1001', '1005', '1001', '1003']
        })
        return df
    
    def test_allocation_basic(self, sample_data):
        """Test basic allocation functionality"""
        result = solve_with_genetic_algorithm(
            sample_data,
            class_size_limit=3,
            max_bullies_per_class=1,
            generations=10,
            pop_size=20
        )
        
        # Check that the result has the expected structure
        assert 'success' in result
        assert result['success'] == True
        assert 'classes' in result
        assert len(result['classes']) > 0
        
        # Check that all students are assigned to a class
        all_students = []
        for class_data in result['classes']:
            all_students.extend([s['id'] for s in class_data['students']])
        
        assert sorted(all_students) == sorted(sample_data['StudentID'].tolist())
    
    def test_constraints_enforcement(self, sample_data):
        """Test that constraints are enforced"""
        # Set strict constraints
        result = solve_with_genetic_algorithm(
            sample_data,
            class_size_limit=2,  # Force at least 3 classes
            max_bullies_per_class=0,  # No bullies allowed per class
            generations=10,
            pop_size=20
        )
        
        # If constraints can't be satisfied completely, there should be violations
        if not result['violations']:
            # If no violations, check that constraints are satisfied
            for class_data in result['classes']:
                # Check class size constraint
                assert len(class_data['students']) <= 2
                
                # Check bullying constraint
                bullies = [s for s in class_data['students'] if s['bullyingScore'] > 7]
                assert len(bullies) == 0

class TestDataProcessing:
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        df = generate_synthetic_data(num_students=10)
        
        # Check basic properties
        assert len(df) == 10
        assert 'StudentID' in df.columns
        assert 'Academic_Performance' in df.columns
        assert 'Wellbeing_Score' in df.columns
        assert 'Bullying_Score' in df.columns
        assert 'Friends' in df.columns
        
        # Check data ranges
        assert df['Academic_Performance'].min() >= 0
        assert df['Academic_Performance'].max() <= 100
        assert df['Wellbeing_Score'].min() >= 1
        assert df['Wellbeing_Score'].max() <= 5
        assert df['Bullying_Score'].min() >= 0
        assert df['Bullying_Score'].max() <= 10
