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

## Getting Started
1. Ensure you have a Python environment and install dependencies (e.g., Flask).  
2. Run the backend server.  
3. Open index.html to begin using ClassForge.  

## License & Contribution
Open for educational discussions and improvements. Feel free to send pull requests or suggestions.

# ClassForge Main Web Interface

**All classroom allocation, visualization, and reporting in this web interface is powered by the genetic algorithm implemented in [`ga.py`](../main/ga.py).**

This folder (`main/`) contains the web-based admin dashboard and user interface for the ClassForge project. It provides interactive tools for uploading data, running allocation algorithms, exploring students and classes, visualizing results, comparing models, making manual overrides, adjusting settings, and exporting reports.

## Model Used

**Note:** The classroom allocation and results shown in this web interface are powered by the genetic algorithm implemented in [`ga.py`](../main/ga.py). All allocation, visualization, and reporting features are based on the output of this genetic algorithm.

## Folder Contents

