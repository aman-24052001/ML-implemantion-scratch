import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    def __init__(self, data, feature_names=None):
        """Initialize with dataset and feature names."""
        self.data = data
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(data.shape[1])]
        self.scaled_data = StandardScaler().fit_transform(data)
        
    def z_score_method(self, threshold=3):
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(self.data))
        outliers = (z_scores > threshold).any(axis=1)
        return outliers, "Z-score"
    
    def iqr_method(self, multiplier=1.5):
        """Detect outliers using IQR method."""
        Q1 = np.percentile(self.data, 25, axis=0)
        Q3 = np.percentile(self.data, 75, axis=0)
        IQR = Q3 - Q1
        outliers = ((self.data < (Q1 - multiplier * IQR)) | 
                   (self.data > (Q3 + multiplier * IQR))).any(axis=1)
        return outliers, "IQR"
    
    def dbscan_method(self, eps=0.5, min_samples=5):
        """Detect outliers using DBSCAN clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(self.scaled_data)
        outliers = clusters == -1
        return outliers, "DBSCAN"
    
    def mahalanobis_distance(self):
        """Detect outliers using Mahalanobis distance."""
        covariance = np.cov(self.scaled_data.T)
        inv_covariance = np.linalg.inv(covariance)
        mean = np.mean(self.scaled_data, axis=0)
        
        distances = []
        for row in self.scaled_data:
            diff = row - mean
            dist = np.sqrt(diff.dot(inv_covariance).dot(diff))
            distances.append(dist)
            
        threshold = np.percentile(distances, 97.5)
        outliers = np.array(distances) > threshold
        return outliers, "Mahalanobis"
    
    def isolation_forest(self, contamination=0.1):
        """Detect outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(self.scaled_data) == -1
        return outliers, "Isolation Forest"
    
    def analyze_outliers(self):
        """Analyze outliers using all methods and create visualizations."""
        methods = [
            self.z_score_method,
            self.iqr_method,
            self.dbscan_method,
            self.mahalanobis_distance,
            self.isolation_forest
        ]
        
        # Collect results
        results = []
        for method in methods:
            outliers, method_name = method()
            num_outliers = np.sum(outliers)
            percent_outliers = (num_outliers / len(outliers)) * 100
            results.append([
                method_name,
                num_outliers,
                f"{percent_outliers:.2f}%",
                np.where(outliers)[0][:5].tolist()  # First 5 outlier indices
            ])
        
        # Create summary table
        summary_table = tabulate(
            results,
            headers=["Method", "Number of Outliers", "Percentage", "Sample Outlier Indices"],
            tablefmt="pretty"
        )
        
        # Create visualizations
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(data=pd.DataFrame(self.data, columns=self.feature_names))
        plt.xticks(rotation=45)
        plt.title("Box Plot of Features")
        
        # 2. Scatter plot matrix (first 3 features)
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            data=pd.DataFrame(self.scaled_data[:, :2], columns=self.feature_names[:2]),
            x=self.feature_names[0],
            y=self.feature_names[1]
        )
        plt.title("Scatter Plot (First 2 Features)")
        
        # 3. Outlier detection comparison
        plt.subplot(2, 2, 3)
        outlier_counts = pd.DataFrame(results, columns=["Method", "Count", "Percentage", "Indices"])
        sns.barplot(x="Method", y="Count", data=outlier_counts)
        plt.xticks(rotation=45)
        plt.title("Number of Outliers by Method")
        
        # 4. Feature distribution
        plt.subplot(2, 2, 4)
        for i in range(min(3, self.data.shape[1])):
            sns.kdeplot(self.data[:, i], label=self.feature_names[i])
        plt.title("Feature Distributions")
        plt.legend()
        
        plt.tight_layout()
        
        return summary_table, fig

# Example usage with different datasets
def analyze_dataset(dataset_name, data, feature_names):
    print(f"\n{'='*50}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*50}")
    
    detector = OutlierDetector(data, feature_names)
    summary, fig = detector.analyze_outliers()
    
    print("\nOutlier Detection Summary:")
    print(summary)
    plt.show()

# Load and analyze different datasets
# California Housing
california = fetch_california_housing()
analyze_dataset("California Housing Dataset", 
               california.data,
               california.feature_names)

# Diabetes Dataset
diabetes = load_diabetes()
analyze_dataset("Diabetes Dataset",
               diabetes.data,
               diabetes.feature_names)

# Breast Cancer Dataset
cancer = load_breast_cancer()
analyze_dataset("Breast Cancer Dataset",
               cancer.data,
               cancer.feature_names)