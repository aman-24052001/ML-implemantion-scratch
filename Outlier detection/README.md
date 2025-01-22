# Outlier Detection Analysis Tool

A comprehensive Python function for detecting and visualizing outliers in datasets using multiple statistical and machine learning methods.

## Features

- **Multiple Detection Methods:**
  - Z-score method (Statistical)
  - IQR (Interquartile Range) method
  - DBSCAN clustering
  - Mahalanobis distance
  - Isolation Forest

- **Visualization Tools:**
  - Box plots for distribution analysis
  - Scatter plots of feature relationships
  - Comparative bar charts of outlier counts
  - Kernel density plots of feature distributions

- **Detailed Summary Statistics:**
  - Number of outliers per method
  - Percentage of data points identified as outliers
  - Sample indices of detected outliers
  - Formatted summary tables

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/outlier-detection.git
cd outlier-detection

# Install required packages
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- tabulate

## Usage

### Basic Usage

```python
from outlier_detector import OutlierDetector
import numpy as np

# Prepare your data
data = np.array([[...], [...], ...])  # Your data matrix
feature_names = ['feature1', 'feature2', ...]  # Your feature names

# Create detector instance
detector = OutlierDetector(data, feature_names)

# Run analysis
summary, fig = detector.analyze_outliers()

# Print results and show visualizations
print(summary)
plt.show()
```

### Example with Built-in Datasets

```python
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
california = fetch_california_housing()

# Analyze dataset
detector = OutlierDetector(california.data, california.feature_names)
summary, fig = detector.analyze_outliers()
```

## Method Details

### Z-Score Method
Identifies outliers based on the number of standard deviations from the mean. Default threshold is 3 standard deviations.

```python
outliers, method_name = detector.z_score_method(threshold=3)
```

### IQR Method
Detects outliers using the Interquartile Range method. Points beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR are considered outliers.

```python
outliers, method_name = detector.iqr_method(multiplier=1.5)
```

### DBSCAN Method
Uses density-based clustering to identify outliers as points that don't belong to any cluster.

```python
outliers, method_name = detector.dbscan_method(eps=0.5, min_samples=5)
```

### Mahalanobis Distance
Identifies outliers based on their distance from the mean, taking into account the covariance structure of the data.

```python
outliers, method_name = detector.mahalanobis_distance()
```

### Isolation Forest
Uses isolation forest algorithm to detect outliers based on the concept of isolation.

```python
outliers, method_name = detector.isolation_forest(contamination=0.1)
```

## Customization

### Modifying Detection Parameters

```python
# Custom Z-score threshold
outliers, _ = detector.z_score_method(threshold=2.5)

# Custom IQR multiplier
outliers, _ = detector.iqr_method(multiplier=2.0)

# Custom DBSCAN parameters
outliers, _ = detector.dbscan_method(eps=0.7, min_samples=10)
```

### Visualization Customization

The `analyze_outliers` method returns both the summary table and the matplotlib figure object, allowing for further customization:

```python
summary, fig = detector.analyze_outliers()
fig.set_size_inches(15, 10)
plt.savefig('outliers.png', dpi=300, bbox_inches='tight')
```

## Output Examples

### Summary Table
```
+------------------+--------------------+------------+----------------------+
| Method           | Number of Outliers | Percentage | Sample Outlier Indices |
+------------------+--------------------+------------+----------------------+
| Z-score          | 127               | 6.35%      | [1, 4, 7, 12, 15]   |
| IQR              | 156               | 7.80%      | [2, 5, 8, 13, 16]   |
| DBSCAN           | 89                | 4.45%      | [3, 6, 9, 14, 17]   |
| Mahalanobis      | 112               | 5.60%      | [4, 7, 10, 15, 18]  |
| Isolation Forest | 100               | 5.00%      | [5, 8, 11, 16, 19]  |
+------------------+--------------------+------------+----------------------+
```

### Visualizations
The tool generates four plots:
1. Box plot showing distribution and outliers
2. Scatter plot of first two features
3. Bar chart comparing outlier counts
4. Kernel density plots showing feature distributions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
