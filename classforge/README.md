# ClassForge

ClassForge is an educational toolkit designed to automate and optimize classroom assignments and groupings for students. It uses algorithms such as Genetic Algorithms with integrated Social Network Analysis (SNA) and Reinforcement Learning to balance academic performance, wellbeing, social cohesion, and other factors.

## Project Structure

`
/classforge
 /backend                # All backend Python code
    /api                # API endpoints
    /algorithms         # Core algorithms (GA, RL, etc.)
    /models             # Data models
    /utils              # Utility functions
    app.py              # Flask application for simplified usage
    main.py             # Main entry point with full API routes
    requirements.txt    # Python dependencies

 /frontend               # All frontend code
    /assets             # Static assets
       /css            # Stylesheets
       /js             # JavaScript files
       /images         # Images
    /pages              # HTML pages
    index.html          # Main entry page

 /data                   # Data files
    /samples            # Sample datasets
    README.md           # Data documentation

 /docs                   # Documentation
    /api                # API documentation
    /user               # User guide
    README.md           # General documentation

 /tests                  # Test suite
    /backend            # Backend tests
    /frontend           # Frontend tests

 .gitignore              # Git ignore file
 README.md               # Project README
 LICENSE                 # Project license
`

## Features

- **Synthetic Data Generation:** Generates and saves student data to CSV if not present.
- **Predictive Modeling:** Estimates risk scores for academic performance, well-being, and peer collaboration.
- **Clustering & Optimization:** Groups students into classes using K-Means, Spectral Clustering, genetic algorithms, and constraint programming.
- **Constraint Programming:** Allocates students to classes with size and academic balance constraints using OR-Tools.
- **Interactive Visualization:** Interface to select clustering method, class, and metric for network coloring; view summary tables, student lists, friendship networks, and metric distributions.

## Installation & Setup

### Backend Setup

1. Navigate to the backend directory:
   `ash
   cd classforge/backend
   `

2. Create a virtual environment:
   `ash
   python -m venv venv
   `

3. Activate the virtual environment:
   - On Windows:
     `ash
     venv\Scripts\activate
     `
   - On macOS/Linux:
     `ash
     source venv/bin/activate
     `

4. Install dependencies:
   `ash
   pip install -r requirements.txt
   `

5. Run the application:
   `ash
   python main.py
   `

### Frontend Setup

1. Navigate to the frontend directory:
   `ash
   cd classforge/frontend
   `

2. Open index.html in a web browser or use a local server:
   `ash
   python -m http.server
   `

## API Documentation

The ClassForge API provides endpoints for:

- Allocating students to classrooms using genetic algorithms
- Generating synthetic student data for testing
- Retrieving and manipulating classroom assignments

For detailed documentation, see the [API Documentation](docs/api/README.md).

## Contributing

1. Fork the repository
2. Create a feature branch: git checkout -b feature-name
3. Commit your changes: git commit -m 'Add some feature'
4. Push to the branch: git push origin feature-name
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
