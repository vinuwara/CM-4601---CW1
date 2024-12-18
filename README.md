Optimizing Nurse Workforce Scheduling Using LSTM and Genetic Algorithms

Overview

This repository contains a hybrid solution for optimizing nurse workforce scheduling, leveraging Long Short-Term Memory (LSTM) networks for demand prediction and Genetic Algorithms (GAs) for schedule optimization. The approach addresses real-world challenges in healthcare scheduling, including balancing demand, minimizing nurse fatigue, and adhering to labor regulations.

Key Features

LSTM Model: Predicts nurse demand based on historical trends, ensuring demand-driven scheduling.

Genetic Algorithm: Optimizes shift assignments by balancing constraints such as workload distribution, maximum shifts per nurse, and legal compliance.

Fitness Function: Combines demand satisfaction and constraint penalties to evaluate scheduling effectiveness.

Visualization Tools: Includes fitness convergence plots, training history, and prediction scatter plots.

Project Structure

.
├── main.py               # Main script to run the optimization.
├── optimizer.py          # Implementation of the AdvancedLSTMGeneticOptimizer class.
├── data/                 # Directory for synthetic or real-world datasets.
├── results/              # Stores generated visualizations (plots).
├── requirements.txt      # List of Python dependencies.
└── README.md             # Project documentation (this file).

Installation

Prerequisites

Python 3.8 or higher

Recommended: Virtual environment for dependency isolation

Steps

Clone the repository:

git clone https://github.com/yourusername/nurse-scheduling-optimizer.git
cd nurse-scheduling-optimizer

Install required dependencies:

pip install -r requirements.txt

Run the main script:

python main.py

Usage

Configurations

Modify the parameters in the main.py file to customize the optimization process:

Problem Dimension: Number of nurses to schedule.

Population Size: Number of individuals in the GA population.

Generations: Maximum number of GA iterations.

Bounds: Range of search space for scheduling variables.

Output

Optimized nurse schedules.

Visualizations saved in the results/ directory:

Fitness convergence plot.

LSTM training history plot.

Prediction vs. actual scatter plot.

Methodology

Synthetic Data Creation: Simulates realistic nurse demand based on patient ratios.

Demand Prediction: Trains an LSTM model to forecast nurse demand.

Schedule Optimization: Uses a GA to refine nurse shift assignments based on LSTM predictions and constraints.

Fitness Function

The fitness function evaluates a schedule using:

Demand Penalty: Measures deviation from predicted nurse demand.

Constraint Penalty: Imposes penalties for constraint violations (e.g., workload imbalance).

Formula:

F = Demand Penalty + λ × Constraint Penalty

Visualizations

Fitness Convergence: Tracks progress of the GA over generations.

Training History: Shows loss reduction during LSTM training.

Prediction Scatter Plot: Compares LSTM predictions to actual values.

Future Enhancements

Incorporate real-world datasets for validation.

Extend to dynamic scheduling scenarios.

Add multi-objective optimization capabilities.
