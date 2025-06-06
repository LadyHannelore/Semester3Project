# Data processing utilities for ClassForge
import pandas as pd
import numpy as np
import random
import networkx as nx

def generate_synthetic_data(num_students=1000, output_file=None):
    """
    Generate synthetic student data for testing and development
    
    Args:
        num_students (int): Number of student records to generate
        output_file (str): Path to save CSV output (optional)
        
    Returns:
        pd.DataFrame: DataFrame with synthetic student data
    """
    # Generate student IDs
    student_ids = [f'{i+1001:04d}' for i in range(num_students)]
    
    # Generate academic performance scores (normal distribution)
    academic_scores = np.random.normal(70, 15, num_students)
    academic_scores = np.clip(academic_scores, 0, 100)
    
    # Generate wellbeing scores (1-5 scale)
    wellbeing_scores = np.random.normal(3.5, 0.8, num_students)
    wellbeing_scores = np.clip(wellbeing_scores, 1, 5)
    
    # Generate bullying scores (1-10 scale, higher = more likely to bully)
    bullying_scores = np.random.exponential(1.5, num_students)
    bullying_scores = np.clip(bullying_scores, 0, 10)
    
    # Generate friendship networks
    friends_list = []
    for i in range(num_students):
        num_friends = np.random.poisson(3)
        num_friends = min(num_friends, 7)  # Cap at 7 friends
        
        friend_candidates = list(range(num_students))
        friend_candidates.remove(i)  # Remove self from candidates
        
        if num_friends > 0:
            friends = random.sample(friend_candidates, num_friends)
            friends_ids = [student_ids[f] for f in friends]
            friends_list.append(','.join(friends_ids))
        else:
            friends_list.append('')
    
    # Create DataFrame
    df = pd.DataFrame({
        'StudentID': student_ids,
        'Academic_Performance': academic_scores,
        'Wellbeing_Score': wellbeing_scores,
        'Bullying_Score': bullying_scores,
        'Friends': friends_list
    })
    
    # Save to file if specified
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df

def build_friendship_graph(df):
    """
    Build a NetworkX graph representing student friendships
    
    Args:
        df (pd.DataFrame): Student data with 'StudentID' and 'Friends' columns
        
    Returns:
        nx.Graph: Graph with students as nodes and friendships as edges
    """
    G = nx.Graph()
    
    # Add all students as nodes
    for _, student in df.iterrows():
        G.add_node(student['StudentID'], 
                  academic=student['Academic_Performance'],
                  wellbeing=student['Wellbeing_Score'],
                  bullying=student['Bullying_Score'])
    
    # Add friendship edges
    for _, student in df.iterrows():
        student_id = student['StudentID']
        if isinstance(student['Friends'], str) and student['Friends'].strip():
            friends = student['Friends'].split(',')
            for friend_id in friends:
                if friend_id.strip():
                    G.add_edge(student_id, friend_id.strip())
    
    return G

def calculate_class_metrics(df, class_column):
    """
    Calculate metrics for each class in the allocation
    
    Args:
        df (pd.DataFrame): Student data with class assignments
        class_column (str): Column name with class assignments
        
    Returns:
        dict: Dictionary with class metrics
    """
    metrics = {}
    
    # Group by class
    class_groups = df.groupby(class_column)
    
    for class_id, group in class_groups:
        metrics[class_id] = {
            'size': len(group),
            'academic_mean': group['Academic_Performance'].mean(),
            'academic_std': group['Academic_Performance'].std(),
            'wellbeing_mean': group['Wellbeing_Score'].mean(),
            'wellbeing_std': group['Wellbeing_Score'].std(),
            'bully_count': sum(group['Bullying_Score'] > 7)
        }
    
    return metrics
