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

## Project Structure
The project is organized as follows:

```
ClassForge/
├── backend/              # Server-side Python code
│   ├── app.py            # Main Flask application
│   ├── models/           # Data models
│   ├── services/         # Business logic including GA
│   │   └── ga.py         # Genetic Algorithm implementation
│   └── utils/            # Helper functions
├── frontend/             # Client-side code
│   ├── assets/
│   │   ├── css/          # Stylesheets
│   │   ├── js/           # JavaScript files
│   │   └── images/       # Image resources
│   ├── pages/            # HTML templates
│   │   ├── allocation.html
│   │   ├── classroom.html
│   │   └── ... 
│   └── index.html        # Main entry point
├── data/                 # Data files
│   ├── synthetic_student_data.csv
│   └── synthetic_student_data_1000.csv
├── docs/                 # Additional documentation
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## File Descriptions

### Backend Components
- **services/ga.py**: Implements the Genetic Algorithm (GA) for classroom allocation with the following functionalities:
  1. **Configuration & Setup**: Defines constants like the number of students, class size targets, and constraints.
  2. **Predictive Analytics**: Uses machine learning models to predict academic success, wellbeing decline, and peer collaboration.
  3. **Genetic Algorithm**: Optimizes classroom assignments by minimizing violations of constraints.
  4. **Flask API**: Provides endpoints for running the allocation algorithm and generating synthetic data.
  5. **Metrics & Violations**: Calculates metrics and identifies constraint violations.

### Frontend Components
- **assets/js/pages/upload.js**: Handles file uploads and data preprocessing for simulation.
- **assets/js/pages/student-explorer.js**: Manages the student explorer interface.
- **assets/js/pages/settings.js**: Handles application settings and configurations.
- **assets/js/script.js**: Initializes the dashboard and manages navigation.
- **assets/js/pages/reports.js**: Generates and exports reports in CSV and PDF formats.
- **assets/js/pages/overrides.js**: Enables manual overrides for student-class assignments.
- **assets/js/pages/classroom.js**: Displays detailed classroom assignments and metrics.
- **assets/js/pages/allocation.js**: Manages the allocation process and algorithm configuration.

### HTML Pages
- **frontend/pages/upload.html**: Interface for uploading student data and running simulations.
- **frontend/pages/student-explorer.html**: Displays student details and allows searching/filtering.
- **frontend/pages/settings.html**: Interface for managing application settings.
- **frontend/pages/reports.html**: Provides options to generate and export reports.
- **frontend/pages/overrides.html**: Interface for manually adjusting student-class assignments.
- **frontend/pages/classroom.html**: Displays classroom assignments and metrics.
- **frontend/pages/allocation.html**: Interface for configuring and running the allocation algorithm.
- **frontend/index.html**: Main dashboard for navigating the application.

### CSS Stylesheets
- **assets/css/styles.css**: Global styles for the application.
- **assets/css/pages/upload.css**: Styles specific to the upload page.
- **assets/css/pages/student-explorer.css**: Styles for the student explorer interface.
- **assets/css/pages/settings.css**: Styles for the settings page.
- **assets/css/pages/reports.css**: Styles for the reports and exports page.
- **assets/css/pages/overrides.css**: Styles for the manual overrides interface.
- **assets/css/pages/classroom.css**: Styles for the classroom view.
- **assets/css/pages/allocation.css**: Styles for the allocation configuration page.

### Data Files
- **data/synthetic_student_data.csv**: Sample dataset for testing the application.
- **data/synthetic_student_data_1000.csv**: Larger sample dataset for scalability testing.

## Theoretical Explanation of the Code

The ClassForge project is a sophisticated system designed to optimize classroom assignments using advanced computational techniques. At its core, the system integrates data-driven analytics, optimization algorithms, and user-friendly interfaces to address the multifaceted challenges of classroom management. The theoretical foundation of the system is rooted in machine learning, genetic algorithms, and constraint satisfaction, ensuring that the solution is both robust and adaptable to diverse educational contexts.

The process begins with **data preparation**, where student data is ingested from CSV files or user inputs. This data typically includes attributes such as academic performance, wellbeing scores, and bullying scores. To enhance the dataset, predictive analytics models, such as XGBoost and Random Forest, are employed to derive additional metrics. These metrics include academic risk, wellbeing risk, and peer collaboration scores, which are critical for understanding the dynamics of student interactions and performance. The preprocessing stage ensures that missing or inconsistent data is handled appropriately, enabling the system to work with real-world datasets that may be incomplete or noisy.

The optimization process is driven by a **Genetic Algorithm (GA)**, a heuristic search method inspired by the principles of natural selection and evolution. The GA operates by iteratively improving a population of candidate solutions, where each solution represents a potential classroom assignment. The algorithm begins with a random initialization of student-to-class assignments. Each solution is evaluated using a fitness function that quantifies how well it satisfies predefined constraints, such as class size limits, academic balance, and bullying distribution. The GA employs evolutionary operators, including selection, crossover, and mutation, to generate new solutions. Over successive generations, the population converges toward an optimal or near-optimal solution. This approach is particularly effective for solving combinatorial optimization problems, where the search space is vast and traditional methods may be computationally infeasible.

A key aspect of the system is its ability to handle **constraints**. These constraints are designed to ensure that the resulting classroom assignments are practical and equitable. For instance, the system enforces a maximum class size to prevent overcrowding and limits the number of high-risk students (e.g., bullies) in each class to maintain a safe learning environment. Additionally, the system strives to balance academic performance and wellbeing across classes, minimizing disparities that could hinder collaborative learning. Violations of these constraints are penalized during the fitness evaluation, guiding the GA toward solutions that adhere to the specified requirements.

The **frontend interface** provides users with an intuitive platform to interact with the system. Users can upload student data, configure optimization parameters, and visualize the results through detailed dashboards. The interface also supports manual overrides, allowing administrators to adjust assignments based on contextual knowledge that may not be captured by the algorithm. This hybrid approach combines the efficiency of automated optimization with the flexibility of human judgment, ensuring that the final assignments are both data-driven and contextually appropriate.

The backend, implemented using Flask, serves as the computational engine of the system. It exposes RESTful APIs for running the optimization algorithm and processing data. The `/allocate` endpoint, for example, accepts student data and optimization parameters, executes the GA, and returns the optimized assignments. This modular architecture allows the system to be easily integrated with other applications or extended with additional features.

The system also includes robust **visualization and reporting** capabilities. Users can generate charts and graphs to analyze the distribution of academic performance, wellbeing, and other metrics across classes. Reports summarize key findings, highlight constraint violations, and provide actionable insights for administrators. These features not only enhance transparency but also facilitate data-driven decision-making.

From a scalability perspective, the system is designed to handle large datasets efficiently. By leveraging parallel processing and optimized algorithms, it can process thousands of students and generate assignments within a reasonable timeframe. This scalability makes it suitable for deployment in diverse educational settings, from small schools to large districts.

In summary, the ClassForge project represents a comprehensive solution to the complex problem of classroom assignment. By combining advanced algorithms, predictive analytics, and user-centric design, it provides a powerful tool for educators and administrators. The system's theoretical foundation ensures that it is both scientifically rigorous and practically effective, making it a valuable asset for modern educational institutions.

## Getting Started
1. Ensure you have a Python environment and install dependencies using `pip install -r requirements.txt`
2. Run the backend server with `python backend/app.py`
3. Open `frontend/index.html` in your browser to begin using ClassForge

## License & Contribution
Open for educational discussions and improvements. Feel free to send pull requests or suggestions.

