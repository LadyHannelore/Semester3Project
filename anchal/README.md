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
