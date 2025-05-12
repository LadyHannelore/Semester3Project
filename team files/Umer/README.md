# ClassForge: AI-Powered Customizable Classroom Allocation

## Project Description

This project, **ClassForge**, is a Python application for exploring classroom allocation using clustering, constraint programming, and optimization. It generates synthetic student data, applies predictive analytics, clusters students into classes, and demonstrates constraint-based allocation using OR-Tools and genetic algorithms. The results are visualized in an interactive Gradio web interface.

---

## Folder Contents and Explanations

### 1. `synthetic_student_data_1000.csv`
- **What it is:** A CSV file containing synthetic (fake but realistic) student data, including academic scores, well-being, social connections, and more.
- **For non-technical users:** This is the "student list" the app uses to simulate a real school.

### 2. `ga.py`
- **What it does:** Implements a genetic algorithm to assign students to classes, aiming for balanced academic performance and social factors. Also provides interactive visualizations (tables, graphs, histograms) using Gradio.
- **For non-technical users:** This is the main engine that tries to create fair and friendly classrooms, and lets you explore the results visually.

### 3. `mine.py`
- **What it does:** Generates synthetic data if not present, preprocesses it, builds a social network graph, and uses multi-objective optimization (NSGA-II) to allocate students to classes. Provides a Gradio interface for users to set priorities (academic, well-being, social) and see trade-offs.
- **For non-technical users:** This lets you experiment with different ways to group students, showing how changing priorities affects classroom fairness and friendships.

### 4. `cp-sat.ipynb`
- **What it does:** A Jupyter notebook demonstrating classroom allocation using Google's OR-Tools CP-SAT solver. Shows how to set up constraints (like class size and academic balance) and solve them step by step.
- **For non-technical users:** This notebook is like a recipe book for the computer, showing how it tries to make classes fair by following certain rules.

### 5. `test.py`
- **What it does:** Contains test code to check that the constraint programming logic works as expected, using a small sample dataset.
- **For non-technical users:** This file is for checking that the "rules" for making classes work correctly.

---

## What This Code Does

- **Synthetic Data Generation:** Creates realistic synthetic student records with academic, well-being, and social attributes.
- **Predictive Analytics:** Uses machine learning models (XGBoost, Random Forest) to estimate academic risk, well-being risk, and peer collaboration scores for each student.
- **Clustering & Optimization:** Assigns students to classes using clustering, genetic algorithms, and constraint programming, balancing academic and social factors.
- **Constraint Programming:** Demonstrates classroom allocation using OR-Tools CP-SAT solver, enforcing constraints on class size and academic balance.
- **Visualization:** Provides interactive tables, network graphs, and histograms to explore class composition and student metrics.

---

## Features

* **Synthetic Data Generation:** Generates and saves student data to CSV if not present.
* **Predictive Modeling:** Estimates risk scores for academic performance, well-being, and peer collaboration.
* **Clustering & Optimization:** Groups students into classes using K-Means, Spectral Clustering, genetic algorithms, and constraint programming.
* **Constraint Programming:** Allocates students to classes with size and academic balance constraints using OR-Tools.
* **Interactive Visualization:** Gradio interface to select clustering method, class, and metric for network coloring; view summary tables, student lists, friendship networks, and metric distributions.

---

## For Non-Technical Users

- **You can use this project to see how computers can help make fair and friendly classrooms.**
- **You can try different settings and see how the groups change.**
- **The visualizations help you understand how students are grouped and how their friendships and academic levels are balanced.**

---

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
