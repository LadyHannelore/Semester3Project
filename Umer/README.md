# ClassForge: AI-Powered Customizable Classroom Allocation

## Project Description

This project, **ClassForge**, is an application I've built using Python to tackle the challenging problem of assigning students to classrooms. The goal is not just to create any assignment, but one that thoughtfully considers multiple factors simultaneously, such as academic performance, student well-being, and social connections. I've used a sophisticated AI technique called multi-objective optimization to find the best possible compromises when these goals conflict. The results are presented through a user-friendly web interface created with Gradio.

## What I Have Done

Based on the code provided, I have implemented the following key components:

1.  **Synthetic Data Generation**: I've created a function to generate realistic-looking synthetic data for a large number of students. This data includes academic scores, well-being indicators (like the K6 scale), and simulated friendships, allowing the system to run without needing real sensitive student information initially. The data is saved to a CSV file for persistence.
2.  **Data Preprocessing**: I've written code to load the synthetic data, normalize academic scores, and build a social graph representing student friendships using the NetworkX library. This graph is crucial for evaluating the social cohesion objective.
3.  **Defined the Optimization Problem**: I've formulated the classroom allocation as a multi-objective optimization problem. This involves defining:
    * **Decision Variables**: How students are assigned to classes. Each student's class assignment is a variable the optimizer can change.
    * **Objectives**: The specific goals to minimize simultaneously (variance in academic performance, variance in well-being scores, and the *negative* of retained friendships, which is the same as maximizing retained friendships).
4.  **Integrated a Multi-Objective Evolutionary Algorithm (NSGA-II)**: I've used the `pymoo` library to apply NSGA-II, a powerful genetic algorithm designed for problems with multiple conflicting objectives.
5.  **Built a Gradio Interface**: I've created an interactive web application using Gradio. This interface allows users to:
    * Specify the desired number of classes.
    * Set priorities (weights) for the different objectives (academic, well-being, social).
    * Run the optimization process.
    * Visualize the **Pareto front**, which shows the trade-offs between the objectives.
    * See a summary of the solution selected based on their weights.
    * View the final student-to-class allocation in a table.

Essentially, I've taken the complex task of balancing multiple student needs in classroom assignments and built an automated system to find and present good compromise solutions using AI.

## Non-Technical Explanation of the NSGA-II Algorithm

Imagine you're trying to pick the best apples from a large orchard, but "best" means different things to different people. Some want the sweetest apples, others want the biggest, and maybe others want apples with the fewest bruises. You can't always find an apple that's the absolute sweetest *and* the absolute biggest *and* has zero bruises. You have to make trade-offs.

This is like **multi-objective optimization**. We have several goals (objectives) that we want to achieve at the same time, but improving one might make another worse.

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is like having a team of smart apple pickers who work together over many rounds to find the best *set* of apples, considering all the different criteria. Here's how it works in simple terms:

1.  **Start Picking (Initial Population)**: The team starts by randomly picking a large basket of apples (this is the initial "population" of possible classroom assignments). Some baskets might be good for sweetness, others for size, etc.
2.  **Evaluate the Apples (Evaluate Objectives)**: For each apple in each basket, they measure how sweet it is, how big it is, and count its bruises (these are like our objective functions – calculating academic variance, well-being variance, and retained friendships for each classroom assignment).
3.  **Sorting by "Goodness" (Non-dominated Sorting)**: Now, they sort the apples (or baskets) based on which ones are clearly better than others. An apple is "better" or **"dominates"** another if it's equal to or better in ALL criteria, and strictly better in at least one. Apples that aren't dominated by *any* other apple are considered part of the "elite front" or **Pareto front**. These represent the best possible trade-offs – you can't improve on one objective without making another worse. Apples on the Pareto front are preferred.
4.  **Handling Ties on the Front (Crowding Distance)**: There might be many apples on the elite front. To encourage variety among these top apples (trade-offs), they also look at how "crowded" an apple is by others similar to it on the front. Apples in less crowded areas are slightly preferred, as they represent more unique trade-off points.
5.  **Picking the Next Generation (Selection)**: Based on this sorting (prioritizing the elite front, then less crowded apples within a front), they select the "parent" apples that will be used to find new apples for the next round.
6.  **Creating New Apples (Crossover and Mutation)**: They combine features from two parent apples (like cross-pollinating, called **crossover**) and introduce small random changes (like a random bruise or extra sweetness, called **mutation**) to create a new batch of "child" apples. This introduces diversity.
7.  **Repeat! (Generations)**: They repeat steps 2-6 many times (these are the "generations"). With each generation, the overall quality of the apples in their baskets improves, and the elite front gets closer to the true best possible trade-offs.
8.  **Final Result**: After many rounds, the algorithm converges, and the final set of apples on the elite front represents the best compromises found across all the criteria. The user can then look at this set (the Pareto front visualization) and decide which specific compromise (which basket of apples/classroom assignment) best fits their overall needs by setting weights.

So, NSGA-II is essentially an intelligent trial-and-error process inspired by evolution, specifically designed to find a whole set of good compromise solutions when you have multiple things you want to achieve simultaneously.

## Features

* **Synthetic Data Generation**: Creates realistic student data with various attributes and social connections.
* **Social Network Modeling**: Builds a graph of student friendships.
* **Multi-Objective Optimization**: Finds optimal classroom assignments balancing academic equity, well-being, and social cohesion.
* **Pareto Front Visualization**: Interactive 3D plot showing the trade-offs between objectives.
* **Weighted Solution Selection**: Choose a preferred allocation from the Pareto front based on your priorities.
* **Detailed Allocation Table**: View the final class assignments and student details.
* **Gradio Web Interface**: User-friendly interface to run the tool and explore results.

## Objectives (Minimized)

1.  Minimize Variance of Average Academic Performance per class.
2.  Minimize Variance of Average K6 Wellbeing Score per class.
3.  Minimize the Negative of Retained Friendships (Maximize Retained Friendships).

## Data

* Synthetic student data is generated in `synthetic_student_data_1000.csv` (filename and student count are configurable).
* Data includes simulated academic performance, well-being scores (K6, PWI), social attitudes, school engagement, language, and friendship connections.
* Friendships are represented as edges in a NetworkX graph.

## Optimization Details

* **Algorithm**: NSGA-II from `pymoo`.
* **Decision Variables**: For `N` students and `K` classes, there are `N` decision variables, each representing the assigned class index (0 to K-1) for a student.
* **Operators**: Integer Random Sampling, Single Point Crossover, Bitflip Mutation.
* **Termination**: Runs for a fixed number of `GENERATIONS`.

## Gradio Interface

The web interface has sections for:

* **Configuration**: Input for the number of classes and sliders for setting the importance weights for Academic Equity, Well-being Balance, and Social Cohesion.
* **Optimization Results**: Displays the interactive 3D Pareto front plot, a summary of the selected solution, and the final classroom allocation table.
* **Status**: A textbox indicating the progress and outcome of the allocation process.

## Installation

To run this project, you need Python and the following libraries. It is recommended to use a virtual environment.

1.  Clone or download the code file (`your_script_name.py`).
2.  Install the required libraries:

    ```bash
    pip install pandas numpy networkx plotly gradio pymoo
    ```

## Usage

1.  Open your terminal or command prompt.
2.  Navigate to the directory where you saved the Python file.
3.  Run the script:

    ```bash
    python app.py
    ```
4.  The script will check for/generate the synthetic data and then launch the Gradio application.
5.  Open the provided local URL (usually `http://127.0.0.1:7860/`) in your web browser.
6.  Configure the number of classes and objective weights, then click "Generate Allocation".
7.  View the Pareto front plot and the details of the selected allocation.

## Configuration

You can modify the following constants at the top of the Python script to change the scale and behavior:

* `NUM_STUDENTS`: Number of synthetic student records to generate.
* `NUM_CLASSES`: Default number of classes to allocate students into.
* `SYNTHETIC_DATA_CSV`: Name of the CSV file for synthetic data.
* `GENERATIONS`: Number of generations for the NSGA-II algorithm. More generations can improve results but take longer.
* `POPULATION_SIZE`: Number of solutions (allocations) in each generation. A larger population explores more options but requires more computation per generation.

## Dependencies

* `os`
* `random`
* `pandas`
* `numpy`
* `networkx`
* `plotly`
* `gradio`
* `pymoo` (including `core.problem`, `algorithms.moo.nsga2`, `operators.sampling.rnd`, `operators.crossover.spx`, `operators.mutation.bitflip`, `optimize`, `termination`)
* `collections`
* `warnings`
