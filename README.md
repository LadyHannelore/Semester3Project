# ClassForge Project

ClassForge is an educational toolkit designed to automate and optimize classroom assignments and groupings for students. It uses algorithms such as Genetic Algorithms with integrated Social Network Analysis (SNA) and Reinforcement Learning to balance academic performance, wellbeing, social cohesion, and other factors.

## Project Overview
- Automatically allocate students to classes based on multiple criteria including academic performance, wellbeing, and social relationships.
- Integrate Social Network Analysis (SNA) to optimize social cohesion within classes using clustering coefficients.
- Provide an interface for exploring classes, students, and allocation results.  
- Allow manual overrides for specific student placements.

## Core Concepts
- "Allocation" assigns students to classes, trying to minimize violations like too many high-risk students in one place while maximizing social cohesion.
- "Social Network Analysis" analyzes friendship networks to ensure students are placed with their social connections when possible.
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

### Backend Components
- **ga.py**: Implements the Genetic Algorithm (GA) for classroom allocation with the following functionalities:
  1. **Configuration & Setup**: Defines constants like the number of students, class size targets, and constraints.
  2. **Predictive Analytics**: Uses machine learning models (XGBoost, Random Forest) to predict academic success, wellbeing decline, and peer collaboration.
  3. **Social Network Analysis (SNA)**: Builds friendship graphs using NetworkX and calculates clustering coefficients to optimize social cohesion within classes.
  4. **Genetic Algorithm**: Optimizes classroom assignments by minimizing violations of constraints while maximizing social connections.
  5. **Flask API**: Provides RESTful endpoints for running the allocation algorithm with `/allocate` and `/` routes.
  6. **Metrics & Violations**: Calculates performance metrics and identifies constraint violations.

### Frontend Components
- **upload.js**: Handles file uploads and data preprocessing for simulation.
- **student-explorer.js**: Manages the student explorer interface.
- **settings.js**: Handles application settings and configurations.
- **script.js**: Initializes the dashboard and manages navigation.
- **overrides.js**: Enables manual overrides for student-class assignments.
- **classroom.js**: Displays detailed classroom assignments and metrics.
- **allocation.js**: Manages the allocation process and algorithm configuration.

### HTML Pages
- **upload.html**: Interface for uploading student data and running simulations.
- **student-explorer.html**: Displays student details and allows searching/filtering.
- **settings.html**: Interface for managing application settings.
- **overrides.html**: Interface for manually adjusting student-class assignments.
- **classroom.html**: Displays classroom assignments and metrics.
- **allocation.html**: Interface for configuring and running the allocation algorithm.
- **index.html**: Main dashboard for navigating the application.

### CSS Stylesheets
- **styles.css**: Global styles for the application.
- **upload.css**: Styles specific to the upload page.
- **student-explorer.css**: Styles for the student explorer interface.
- **settings.css**: Styles for the settings page.
- **overrides.css**: Styles for the manual overrides interface.
- **classroom.css**: Styles for the classroom view.
- **allocation.css**: Styles for the allocation configuration page.
- **dashboard.css**: Styles for the main dashboard interface.

### Data Files
- **synthetic_student_data.csv**: Sample dataset for testing the application.
- **synthetic_student_data_1000.csv**: Larger sample dataset for scalability testing.
- **requirements.txt**: Python dependencies required for the backend server.

### Additional Files
- **Project_Report_Template.md**: Template for project documentation and reporting.
- **Sem_Presentation_G11.pptx**: Project presentation slides.

## Theoretical Explanation of the Code

The ClassForge project is a sophisticated system designed to optimize classroom assignments using advanced computational techniques. At its core, the system integrates data-driven analytics, optimization algorithms, Social Network Analysis (SNA), and user-friendly interfaces to address the multifaceted challenges of classroom management. The theoretical foundation of the system is rooted in machine learning, genetic algorithms, graph theory, and constraint satisfaction, ensuring that the solution is both robust and adaptable to diverse educational contexts.

The process begins with **data preparation**, where student data is ingested from CSV files or user inputs. This data typically includes attributes such as academic performance, wellbeing scores, bullying scores, and friendship networks. To enhance the dataset, predictive analytics models, such as XGBoost and Random Forest, are employed to derive additional metrics. These metrics include academic risk, wellbeing risk, and peer collaboration scores, which are critical for understanding the dynamics of student interactions and performance. The preprocessing stage ensures that missing or inconsistent data is handled appropriately, enabling the system to work with real-world datasets that may be incomplete or noisy.

**Social Network Analysis (SNA)** is a key innovation in this system. The friendship data is used to construct a social graph using NetworkX, where students are nodes and friendships are edges. The system calculates clustering coefficients for each class assignment to measure social cohesion. A higher clustering coefficient indicates that students within a class have more interconnected friendships, promoting better social dynamics and peer support. The SNA component penalizes assignments that break up friend groups or result in low social cohesion within classes.

The optimization process is driven by a **Genetic Algorithm (GA)**, a heuristic search method inspired by the principles of natural selection and evolution. The GA operates by iteratively improving a population of candidate solutions, where each solution represents a potential classroom assignment. The algorithm begins with a random initialization of student-to-class assignments. Each solution is evaluated using a comprehensive fitness function that quantifies how well it satisfies predefined constraints, including:

- Class size limits and balance
- Academic performance distribution
- Bullying score limitations
- Wellbeing score constraints  
- Friendship preservation (social penalty)
- Social cohesion optimization (SNA penalty)

The GA employs evolutionary operators, including selection, crossover, and mutation, to generate new solutions. Over successive generations, the population converges toward an optimal or near-optimal solution. This approach is particularly effective for solving combinatorial optimization problems, where the search space is vast and traditional methods may be computationally infeasible.

A key aspect of the system is its ability to handle **multiple constraints simultaneously**. These constraints are designed to ensure that the resulting classroom assignments are practical, equitable, and socially beneficial. For instance, the system enforces a maximum class size to prevent overcrowding and limits the number of high-risk students (e.g., bullies) in each class to maintain a safe learning environment. Additionally, the system strives to balance academic performance and wellbeing across classes while maximizing social connections, minimizing disparities that could hinder collaborative learning. Violations of these constraints are penalized during the fitness evaluation, guiding the GA toward solutions that adhere to the specified requirements.

The **frontend interface** provides users with an intuitive platform to interact with the system. Users can upload student data, configure optimization parameters, and visualize the results through detailed dashboards. The interface also supports manual overrides, allowing administrators to adjust assignments based on contextual knowledge that may not be captured by the algorithm. This hybrid approach combines the efficiency of automated optimization with the flexibility of human judgment, ensuring that the final assignments are both data-driven and contextually appropriate.

The backend, implemented using Flask, serves as the computational engine of the system. It exposes RESTful APIs for running the optimization algorithm and processing data. The `/allocate` endpoint, for example, accepts student data and optimization parameters, executes the GA with SNA integration, and returns the optimized assignments along with social cohesion metrics. This modular architecture allows the system to be easily integrated with other applications or extended with additional features.

The system also includes robust **visualization and reporting** capabilities. Users can generate charts and graphs to analyze the distribution of academic performance, wellbeing, social connections, and other metrics across classes. Reports summarize key findings, highlight constraint violations, and provide actionable insights for administrators. These features not only enhance transparency but also facilitate data-driven decision-making.

From a scalability perspective, the system is designed to handle large datasets efficiently. By leveraging parallel processing, optimized algorithms, and efficient graph operations, it can process thousands of students and their social networks to generate assignments within a reasonable timeframe. This scalability makes it suitable for deployment in diverse educational settings, from small schools to large districts.

In summary, the ClassForge project represents a comprehensive solution to the complex problem of classroom assignment. By combining advanced algorithms, predictive analytics, social network analysis, and user-centric design, it provides a powerful tool for educators and administrators. The system's theoretical foundation ensures that it is both scientifically rigorous and practically effective, making it a valuable asset for modern educational institutions seeking to optimize both academic outcomes and social well-being.

## Getting Started
1. Ensure you have a Python environment and install dependencies using `pip install -r requirements.txt`
2. Run the backend server with `python ga.py`
3. Open `index.html` in your browser to begin using ClassForge

## Key Features
- **Multi-constraint optimization**: Balances academic performance, wellbeing, social connections, and safety constraints
- **Social Network Analysis**: Uses friendship data to optimize class social cohesion through clustering coefficient calculations
- **Machine Learning integration**: Employs XGBoost and Random Forest for predictive analytics
- **Interactive web interface**: User-friendly dashboard for data upload, configuration, and result visualization
- **Manual override capabilities**: Allows administrators to make contextual adjustments
- **Comprehensive reporting**: Generates detailed metrics and constraint violation reports

## License & Contribution
Open for educational discussions and improvements. Feel free to send pull requests or suggestions.

