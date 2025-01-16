# Dimensionality Reduction: PCA vs LDA Implementation

This repository contains a Python implementation of two popular dimensionality reduction techniques: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). The implementation is done from scratch using NumPy and demonstrates the key differences between these approaches.

## Table of Contents
- [Overview](#overview)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

Both PCA and LDA are dimensionality reduction techniques, but they serve different purposes and are used in different scenarios:

- **PCA**: Unsupervised learning technique focused on variance preservation
- **LDA**: Supervised learning technique focused on class separability

## Principal Component Analysis (PCA)

### What is PCA?
PCA is an unsupervised dimensionality reduction technique that transforms high-dimensional data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they explain in the data.

### Why Use PCA?
- Reduce dimensionality while preserving maximum variance
- Remove multicollinearity
- Feature extraction and selection
- Data visualization
- Noise reduction

### PCA Implementation Steps

1. **Data Standardization**
   - Convert features to same scale
   - Make mean = 0 and standard deviation = 1
   - Formula: Z = (X - μ) / σ

2. **Covariance Matrix Calculation**
   - Compute covariance between all pairs of features
   - Formula: Cov(X,Y) = Σ((X - μx)(Y - μy)) / (n-1)

3. **Eigenvalue and Eigenvector Computation**
   - Find eigenvalues and eigenvectors of covariance matrix
   - Eigenvalues represent variance explained
   - Eigenvectors represent direction of principal components

4. **Principal Component Selection**
   - Sort eigenvalues in descending order
   - Select top k eigenvectors based on:
     - Desired number of dimensions
     - Cumulative variance explained threshold

5. **Data Transformation**
   - Project original data onto new principal component axes
   - Result is reduced-dimensional data

## Linear Discriminant Analysis (LDA)

### What is LDA?
LDA is a supervised dimensionality reduction technique that finds a linear combination of features that characterizes or separates two or more classes while maintaining class-discriminatory information.

### Why Use LDA?
- Maximize class separability
- Reduce within-class variance
- Feature extraction for classification
- Supervised dimensionality reduction
- Pre-processing step for classification tasks

### LDA Implementation Steps

1. **Data Standardization**
   - Similar to PCA, standardize features
   - Essential for comparing features on same scale

2. **Within-Class Scatter Matrix (Sw)**
   - Compute variance within each class
   - Measures how features vary within classes
   - Formula: Sw = Σ Σ(x - μk)(x - μk)ᵀ

3. **Between-Class Scatter Matrix (Sb)**
   - Compute variance between class means
   - Measures separation between classes
   - Formula: Sb = Σ Nk(μk - μ)(μk - μ)ᵀ

4. **Eigenvalue Decomposition**
   - Solve eigenvalue problem: Sw⁻¹Sb
   - Find eigenvectors that maximize between-class variance
   - While minimizing within-class variance

5. **Linear Discriminants Selection**
   - Choose top k eigenvectors
   - Usually k = number of classes - 1
   - Transform data using selected vectors

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dimensionality-reduction.git

# Install required packages
pip install numpy pandas matplotlib scikit-learn
```

## Usage

```python
from dimension_reducer import PCA, LDA

# Initialize reducers
pca = PCA(n_components=2)
lda = LDA(n_components=2)

# Fit and transform data
X_pca = pca.fit_transform(X_scaled)
X_lda = lda.fit_transform(X_scaled, y)
```

## Results

The implementation includes visualization utilities that show:
- PCA projection with variance explained percentages
- LDA projection showing class separation
- Comparative analysis between both techniques

### Key Differences

1. **Supervision**
   - PCA: Unsupervised, doesn't use class labels
   - LDA: Supervised, requires class labels

2. **Optimization Objective**
   - PCA: Maximizes variance in data
   - LDA: Maximizes class separation

3. **Use Cases**
   - PCA: General dimensionality reduction, feature extraction
   - LDA: Pre-processing for classification tasks

4. **Components Limit**
   - PCA: Can extract up to min(n_samples, n_features) components
   - LDA: Limited to n_classes - 1 components

## Thankyou
