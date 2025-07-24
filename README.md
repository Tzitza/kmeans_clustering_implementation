# K-means Clustering Implementation

A Python implementation of the K-means clustering algorithm with Euclidean distance, featuring step-by-step visualization and SSE convergence tracking.

## Overview

This project implements the K-means clustering algorithm from scratch using Python, developed as part of a Data Mining course assignment at the University of Ioannina. The implementation includes visualization of each clustering step and analysis of the Sum of Squared Errors (SSE) convergence.

## Features

- **Custom K-means Implementation**: Built from scratch using only NumPy for mathematical operations
- **Euclidean Distance**: Uses Euclidean distance metric for cluster assignment
- **Step-by-step Visualization**: Shows clustering progress at each iteration
- **SSE Tracking**: Monitors and plots Sum of Squared Errors convergence
- **Multi-variate Normal Data Generation**: Creates test data from three different normal distributions

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

```bash
pip install numpy matplotlib
```

## Usage

Run the main script:

```bash
python k-means.py
```

## Algorithm Details

### Input Parameters
- **Data**: N×M matrix where N is the number of data points and M is the number of features
- **K**: Number of clusters
- **epsilon**: Convergence threshold (default: 1e-3)
- **max_iter**: Maximum number of iterations (default: 100)

### Output
- **ClusterCenters**: K×M matrix containing the final cluster centers
- **IDC**: N×1 vector with cluster labels for each data point
- **SSE_list**: List of SSE values for each iteration

### Test Data Specifications

The algorithm is tested on 150 2D points generated from three multivariate normal distributions:

- **Distribution 1**: μ₁ = [4, 0], Σ₁ = [[0.29, 0.4], [0.4, 4]]
- **Distribution 2**: μ₂ = [5, 7], Σ₂ = [[0.29, 0.4], [0.4, 0.9]]  
- **Distribution 3**: μ₃ = [7, 4], Σ₃ = [[0.64, 0], [0, 0.64]]

Each distribution contributes 50 points to the dataset.

## Visualization

The implementation provides three types of visualizations:

- **Step-by-step Clustering**: Shows data points colored by cluster assignment and centers marked with '+' symbols for each iteration
- **Final Clustering**: Displays the final clustering result
- **SSE Convergence**: Plots the Sum of Squared Errors across iterations

## Algorithm Flow

1. **Initialization**: Randomly select K initial centers from the data points
2. **Assignment**: Assign each point to the nearest cluster center using Euclidean distance
3. **Update**: Recalculate cluster centers as the mean of assigned points
4. **Convergence Check**: Stop if center movement is below epsilon threshold
5. **Visualization**: Display current clustering state
6. **Repeat**: Continue until convergence or maximum iterations reached

## Functions

### Core Functions
- `mykmeans(data, k, epsilon, max_iter)`: Main K-means implementation
- `generate_data()`: Creates test dataset from specified normal distributions

### Visualization Functions
- `plot_kmeans_step(data, labels, centers, step)`: Plots clustering state at each step
- `plot_final_clustering(data, labels, centers)`: Shows final clustering result
- `plot_sse(sse_list)`: Displays SSE convergence graph

## Implementation Notes

- **Random Initialization**: Centers are randomly selected from existing data points (no replacement)
- **Convergence Criteria**: Algorithm stops when center movement is less than epsilon
- **Color Scheme**: Uses red, green, and blue for the three clusters
- **Center Markers**: Black '+' symbols mark cluster centers
- **Time-based Seeding**: Uses current time for random seed to ensure different results each run

## Results

The algorithm successfully clusters the generated data into three distinct groups, with visualizations showing:
- Clear separation of the three underlying normal distributions
- Convergence of cluster centers to optimal positions
- Decreasing SSE values across iterations until convergence

## Assignment Details

- **Course**: Data Mining (Εξόρυξη Δεδομένων)
- **Institution**: Department of Computer Science and Telecommunications, University of Ioannina
- **Submission Date**: January 5, 2025
- **Student ID**: ΤΖΙΤΖΑ_2589

## License

This project is developed for educational purposes as part of a university assignment.
