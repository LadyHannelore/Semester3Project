# ClassForge: AI-Powered Customizable Classroom Allocation

## Project Description

This project, **ClassForge**, is a Python application for exploring classroom allocation using clustering and constraint programming. It generates synthetic student data, applies predictive analytics, clusters students into classes using K-Means and Spectral Clustering, and demonstrates constraint-based allocation using OR-Tools. The results are visualized in an interactive Gradio web interface.

## What This Code Does

- **Synthetic Data Generation**: Creates realistic synthetic student records with academic, well-being, and social attributes.
- **Predictive Analytics**: Uses machine learning models (XGBoost, Random Forest) to estimate academic risk, well-being risk, and peer collaboration scores for each student.
- **Clustering**: Assigns students to classes using K-Means and Spectral Clustering based on their features and predicted risks.
- **Constraint Programming**: Demonstrates classroom allocation using OR-Tools CP-SAT solver, enforcing constraints on class size and academic balance.
- **Visualization**: Provides interactive tables, network graphs, and histograms to explore class composition and student metrics.

## Features

* **Synthetic Data Generation**: Generates and saves student data to CSV if not present.
* **Predictive Modeling**: Estimates risk scores for academic performance, well-being, and peer collaboration.
* **Clustering**: Groups students into classes using K-Means and Spectral Clustering.
* **Constraint Programming**: Allocates students to classes with size and academic balance constraints using OR-Tools.
* **Interactive Visualization**: Gradio interface to select clustering method, class, and metric for network coloring; view summary tables, student lists, friendship networks, and metric distributions.

## Data

* Synthetic data is generated in `synthetic_student_data.csv`.
* Each student has academic, well-being, social, and friendship attributes.
* Friendships are represented as comma-separated lists of student IDs.

## Gradio Interface

The web interface allows you to:

* Select the clustering method (K-Means or Spectral).
* Choose a class ID to view.
* Select a metric to color the friendship network.
* View summary statistics for all classes.
* See a table of students in the selected class.
* Visualize the friendship network within the class, colored by the chosen metric.
* View histograms of academic and well-being risk for the selected class.

## Installation

To run this project, you need Python and the following libraries. It is recommended to use a virtual environment.

1.  Clone or download the code file (`ga.py`).
2.  Install the required libraries:

    ```bash
    conda create -n classforge310 python=3.10
    conda activate classforge310
    pip install ortools gradio xgboost scikit-learn matplotlib networkx pandas numpy
    ```

## Usage

1.  Open your terminal or command prompt.
2.  Navigate to the directory where you saved the Python file.
3.  Run the script:

    ```bash
    python ga.py
    ```
4.  The script will generate synthetic data if needed and launch the Gradio application.
5.  Open the provided local URL (usually `http://127.0.0.1:7860/`) in your web browser.
6.  Use the dropdowns to explore clustering/class assignments and student metrics.

## Configuration

You can modify the following constants at the top of the Python script to change the scale and behavior:

* `NUM_STUDENTS`: Number of synthetic student records to generate.
* `CLASS_SIZE_TARGET`: Target class size for clustering.
* `SYNTHETIC_DATA_CSV`: Name of the CSV file for synthetic data.

## Dependencies

* `os`
* `random`
* `pandas`
* `numpy`
* `networkx`
* `matplotlib`
* `gradio`
* `xgboost`
* `scikit-learn`
* `ortools`
* `warnings`
