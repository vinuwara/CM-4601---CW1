# Optimizing Nurse Workforce Scheduling Using LSTM and Genetic Algorithms

## Overview
This repository contains a hybrid solution for optimizing nurse workforce scheduling, leveraging Long Short-Term Memory (LSTM) networks for demand prediction and Genetic Algorithms (GAs) for schedule optimization. The approach addresses real-world challenges in healthcare scheduling, including balancing demand, minimizing nurse fatigue, and adhering to labor regulations.

---

## Key Features
- **LSTM Model:** Predicts nurse demand based on historical trends, ensuring demand-driven scheduling.
- **Genetic Algorithm:** Optimizes shift assignments by balancing constraints such as workload distribution, maximum shifts per nurse, and legal compliance.
- **Fitness Function:** Combines demand satisfaction and constraint penalties to evaluate scheduling effectiveness.
- **Visualization Tools:** Includes fitness convergence plots, training history, and prediction scatter plots.

---

## Project Structure
```plaintext
     
├── main_GeneticAlgo_LSTM.py    # main code / impelentation.
├── diagrams/                   # Stores generated visualizations and diagrams.
├── requirements.txt            # List of Python dependencies.
└── README.md                   # Project documentation (this file).
```
## Usage
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
