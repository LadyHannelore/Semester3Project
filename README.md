# ClassForge Project

ClassForge is an educational toolkit designed to automate and optimize classroom assignments and groupings for students. It uses algorithms such as Genetic Algorithms and Reinforcement Learning to balance academic performance, wellbeing, and other factors.

## Project Overview
- Automatically allocate students to classes based on multiple criteria.  
- Provide an interface for exploring classes, students, and allocation results.  
- Allow manual overrides for specific student placements.

## Core Concepts
- "Allocation" assigns students to classes, trying to minimize violations like too many high-risk students in one place.  
- "Explorer" shows detailed information about each student and class.  
- "Reporting" helps you export results.

## For Non-Technical Users
1. Go to “Upload & Simulate” to load your student data (CSV or generated).  
2. Click “Group Allocation” to run an algorithm that places students into classes automatically.  
3. Review details in “Classroom View” and make any overrides as needed.  

## Technical Notes
- The backend (Flask server) processes optimization requests.  
- A separate script runs advanced ML or GA approaches.  

## File Descriptions

### Python Files
- **ga.py**: Implements the Genetic Algorithm for classroom allocation. Includes constraints for academic balance, bullying, and wellbeing.

### JavaScript Files
- **upload.js**: Handles file uploads and data preprocessing for simulation.
- **student-explorer.js**: Manages the student explorer interface, allowing users to view and search student details.
- **settings.js**: Handles application settings and configurations.
- **script.js**: Initializes the dashboard and manages navigation.
- **reports.js**: Generates and exports reports in CSV and PDF formats.
- **overrides.js**: Enables manual overrides for student-class assignments.
- **classroom.js**: Displays detailed classroom assignments and metrics.
- **allocation.js**: Manages the allocation process, including parameter configuration and running the Genetic Algorithm.

### HTML Files
- **upload.html**: Interface for uploading student data and running simulations.
- **student-explorer.html**: Displays student details and allows searching/filtering.
- **settings.html**: Interface for managing application settings.
- **reports.html**: Provides options to generate and export reports.
- **overrides.html**: Interface for manually adjusting student-class assignments.
- **classroom.html**: Displays classroom assignments and metrics.
- **allocation.html**: Interface for configuring and running the allocation algorithm.
- **index.html**: Main dashboard for navigating the application.

### CSS Files
- **styles.css**: Global styles for the application.
- **upload.css**: Styles specific to the upload page.
- **student-explorer.css**: Styles for the student explorer interface.
- **settings.css**: Styles for the settings page.
- **reports.css**: Styles for the reports and exports page.
- **overrides.css**: Styles for the manual overrides interface.
- **classroom.css**: Styles for the classroom view.
- **allocation.css**: Styles for the allocation configuration page.

### Data Files
- **synthetic_student_data.csv**: Sample dataset for testing the application.
- **synthetic_student_data_1000.csv**: Larger sample dataset for scalability testing.

## Getting Started
1. Ensure you have a Python environment and install dependencies (e.g., Flask).  
2. Run the backend server.  
3. Open index.html to begin using ClassForge.  

## License & Contribution
Open for educational discussions and improvements. Feel free to send pull requests or suggestions.

