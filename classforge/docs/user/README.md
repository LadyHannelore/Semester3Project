# ClassForge User Guide

Welcome to ClassForge, an AI-powered tool for optimizing classroom allocations. This guide will help you understand how to use the system effectively.

## Getting Started

1. **Launch ClassForge**: Open the index.html file in your web browser or access the hosted version.

2. **Navigate the Dashboard**: The main dashboard provides access to all features:
   - Upload & Simulate: Load or generate student data
   - Group Allocation: Run the allocation algorithm
   - Classroom View: Explore the generated classrooms
   - Student Explorer: View and filter individual students
   - Manual Overrides: Make changes to automatic allocations
   - Settings: Configure system parameters

## Key Features

### Data Management

The system can work with:
- Existing student data in CSV format
- Synthetic data generated for testing

To upload your own data, use the 'Upload & Simulate' page. Your data should have the following columns:
- StudentID: Unique identifier for each student
- Academic_Performance: Academic scores (0-100)
- Wellbeing_Score: Student wellbeing metric (1-5)
- Bullying_Score: Risk assessment for bullying behavior (0-10)
- Friends: Comma-separated list of StudentIDs representing friendships

### Classroom Allocation

1. Navigate to 'Group Allocation'
2. Select the algorithm (Genetic Algorithm is recommended)
3. Configure parameters:
   - Class Size: Maximum number of students per class
   - Bullying Constraint: Maximum high-risk students per class
   - Wellbeing Range: Optional min/max for wellbeing scores
   - Generation Count: Number of iterations (higher = better results but slower)
   - Population Size: Size of solution population (higher = more diversity)
4. Click 'Run Allocation' to generate classroom assignments

### Exploring Results

After running an allocation:

1. **Classroom View**:
   - Select a class from the dropdown menu
   - View academic and wellbeing metrics
   - Explore the friendship network graph
   - See detailed student lists

2. **Student Explorer**:
   - Search for specific students
   - Filter by academic performance, wellbeing, or risk factors
   - View individual student details and social connections

3. **Manual Overrides**:
   - Move students between classes
   - Prioritize specific friendship connections
   - Address any constraint violations identified by the system

## Understanding the Algorithms

ClassForge uses several advanced algorithms:

1. **Genetic Algorithm**: Optimizes classroom assignments by balancing multiple objectives:
   - Maintaining balanced academic performance across classes
   - Keeping friends together when possible
   - Distributing high-risk students
   - Maintaining wellbeing balance

2. **Social Network Analysis**: Analyzes friendship networks to ensure social cohesion in each classroom.

3. **Predictive Analytics**: Uses machine learning models to identify potential academic or wellbeing risks.

## Troubleshooting

- **Slow Performance**: For large datasets (1000+ students), allocation may take several minutes. Reduce 'Generation Count' and 'Population Size' for faster results.
- **Friendship Violations**: Not all friendship connections can be maintained while respecting other constraints. Use the Manual Overrides to prioritize critical relationships.
- **Browser Compatibility**: For best results, use the latest version of Chrome, Firefox, or Edge.

## Getting Help

For additional support, please refer to:
- Project README: More technical details about the system
- API Documentation: For developers integrating with the system
