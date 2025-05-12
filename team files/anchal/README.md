# Student Classroom Assignment with GNNs

This project uses Graph Neural Networks (GNNs) and clustering to assign students to classrooms while considering academic performance, wellbeing, friendliness, and bullying risk. The workflow is implemented in a Jupyter notebook and leverages PyTorch Geometric, scikit-learn, and pandas.

## Features

- Loads and preprocesses synthetic student data.
- Constructs a friendship network graph.
- Builds node features from academic, wellbeing, and social metrics.
- Applies a GCN (Graph Convolutional Network) to learn student embeddings.
- Uses KMeans clustering to assign students to classrooms.
- Evaluates classroom assignments against constraints (academic balance, wellbeing, bullying spread, class size).
- Visualizes results and provides summary statistics.

## Setup

1. Clone this repository or download the files.
2. Install required Python packages:
   ```bash
   pip install torch torch-geometric scikit-learn pandas networkx matplotlib
   ```
3. Ensure the `synthetic_student_data.csv` file is present in the folder.

## Usage

Open the notebook `Workbook_with_full_constraints[1].ipynb` in Jupyter or VS Code and run the cells sequentially. The notebook will guide you through data loading, processing, model training, clustering, and evaluation.

### How the Code Works (Non-Technical Overview)

- **Data Loading and Preparation:**  
  The notebook starts by loading a file containing synthetic student data. It then calculates new scores for each student, such as wellbeing, friendliness, and bullying risk, based on the original data. These scores are normalized so they can be compared fairly.

- **Building the Friendship Network:**  
  Each student lists their friends. The code uses this information to build a network (graph) where each student is a node, and friendships are the connections (edges) between them.

- **Feature Engineering:**  
  For each student, the code creates a set of features (numbers) that describe their academic performance, wellbeing, friendliness, bullying risk, and a combined score. These features are used to represent each student in the network.

- **Graph Neural Network (GNN) Modeling:**  
  The code uses a type of artificial intelligence called a Graph Neural Network (GNN). This model learns to represent each student in a way that takes into account both their own features and their friendships. The result is a set of "embeddings"—specialized coordinates for each student that capture both their characteristics and their social context.

- **Clustering Students into Classrooms:**  
  Using the learned embeddings, the code groups students into classrooms using a clustering algorithm. The goal is to create classrooms that are balanced in terms of academic performance, wellbeing, and social factors, while also spreading out students who may be at risk of bullying.

- **Constraint Evaluation:**  
  After assigning students to classrooms, the code checks if the assignments meet certain requirements. For example, it checks that no classroom has too many students at risk of bullying, that academic and wellbeing scores are balanced across classrooms, and that class sizes are reasonable.

- **Visualization and Reporting:**  
  The notebook includes charts and tables to help visualize the results, such as how students are grouped, the average scores in each classroom, and whether the constraints are satisfied.

- **Summary:**  
  This workflow helps demonstrate how advanced AI techniques can be used to create fair and balanced classroom assignments, taking into account both academic and social factors.

## Files

- `Workbook_with_full_constraints[1].ipynb`: Main notebook with code and analysis.
- `synthetic_student_data.csv`: Input data file (not included here).
- `README.md`: Project overview and instructions.

## Notes

- The notebook is designed for demonstration and experimentation.
- Adjust the number of classrooms or constraint thresholds as needed.
- Results may vary with different random seeds or data.

## License

For academic and research use only.
