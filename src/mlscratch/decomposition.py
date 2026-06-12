"""
PCA from scratch via the covariance-matrix eigendecomposition (HW1) — the transform
sklearn.decomposition.PCA gives you, spelled out.
"""
from __future__ import annotations

import numpy as np


def pca_algorithm(dataset, k):
    # finding mean and standard deviation value of each feature
    features_mean = np.mean(dataset, axis=0)
    features_std = np.std(dataset, axis=0)

    # centering/standardizing wine features data
    features_scale = (dataset - features_mean) / features_std

    # covariance matrix computation
    n_rows = features_scale.shape[0]
    features_cov_matrix = (features_scale.T @ features_scale) / (n_rows - 1)

    # Eigenvalue/eigenvector calculation
    # using np.linalg.eigh to calculate eigen vectors and values from a symmetric covariance matrix
    eigen_result = np.linalg.eigh(features_cov_matrix)
    eigenvalues = eigen_result[0] # storing eigenvalues
    eigenvectors = eigen_result[1] # storing eigenvectors

    # sorting the results of np.linalg.eigh from largest to smallest eigenvalues and rearranging columns of eigenvectors matrix to match them with the order of eigenvalues
    sorted_eigenvalues_indices = np.argsort(eigenvalues)[::-1] # this gives the indices of descendingly sorted eigenvalues

    # sorting both eigenvectors and eigenvalues using the indices array from np.argsort from largest to smallest
    sorted_eigenvalues = eigenvalues[sorted_eigenvalues_indices]
    sorted_eigenvectors = eigenvectors[:,sorted_eigenvalues_indices]

    # Projection of data onto principal components
    # first we will create a projection matrix made of top k features
    projection_matrix = sorted_eigenvectors[:,:k] # selecting all rows of first k columns
    projected_data = features_scale @ projection_matrix

    # Variance explained calculation
    total_variance = np.sum(eigenvalues) # total sum of all eigen values
    explained_variance_ratio = sorted_eigenvalues / total_variance

    return projected_data, explained_variance_ratio
