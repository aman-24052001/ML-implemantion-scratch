import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DimensionalityReducer:
    """Base class for dimensionality reduction techniques."""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
    
    def fit(self, X):
        """Fit the model with X."""
        raise NotImplementedError
    
    def transform(self, X):
        """Apply dimensionality reduction to X."""
        raise NotImplementedError
    
    def fit_transform(self, X):
        """Fit the model with X and apply dimensionality reduction."""
        self.fit(X)
        return self.transform(X)

class PCA(DimensionalityReducer):
    """Principal Component Analysis implementation."""
    
    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components and explained variance ratio
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

class LDA(DimensionalityReducer):
    """Linear Discriminant Analysis implementation."""
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Compute class means and overall mean
        self.means_ = []
        X_by_class = []
        
        for cls in self.classes_:
            X_cls = X[y == cls]
            self.means_.append(np.mean(X_cls, axis=0))
            X_by_class.append(X_cls)
            
        self.overall_mean_ = np.mean(X, axis=0)
        
        # Compute between-class scatter matrix
        self.means_ = np.array(self.means_)
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean in enumerate(self.means_):
            n_samples = len(X_by_class[i])
            mean_diff = mean - self.overall_mean_
            S_B += n_samples * np.outer(mean_diff, mean_diff)
            
        # Compute within-class scatter matrix
        S_W = np.zeros_like(S_B)
        for i, X_cls in enumerate(X_by_class):
            mean = self.means_[i]
            for sample in X_cls:
                diff = sample - mean
                S_W += np.outer(diff, diff)
                
        # Compute eigenvectors and eigenvalues of S_W^-1 * S_B
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        self.components_ = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        return np.dot(X - self.overall_mean_, self.components_)

def visualize_results(X_pca, X_lda, y, explained_variance_ratio=None):
    """Visualize PCA and LDA results side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PCA results
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    ax1.set_title('PCA Projection')
    ax1.set_xlabel(f'First Principal Component\n{explained_variance_ratio[0]:.2%} variance explained')
    ax1.set_ylabel(f'Second Principal Component\n{explained_variance_ratio[1]:.2%} variance explained')
    
    # Plot LDA results
    scatter2 = ax2.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
    ax2.set_title('LDA Projection')
    ax2.set_xlabel('First Discriminant')
    ax2.set_ylabel('Second Discriminant')
    
    plt.colorbar(scatter1, ax=ax1, label='Class')
    plt.colorbar(scatter2, ax=ax2, label='Class')
    plt.tight_layout()
    return fig

def main():
    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply LDA
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Visualize results
    fig = visualize_results(X_pca, X_lda, y, pca.explained_variance_ratio_)
    plt.show()
    
    return pca, lda, X_pca, X_lda

if __name__ == "__main__":
    pca, lda, X_pca, X_lda = main()