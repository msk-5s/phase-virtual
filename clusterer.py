# SPDX-License-Identifier: MIT

"""
This module contains functions for clustering.

Functions in the module must adhere to the following signature:
def function(data: NDArray[(Any, Any), float], random_state: int) -> NDArray[(Any, Any), int]
"""

from typing import Any
from nptyping import NDArray

import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import sklearn.mixture
import sklearn_extra.cluster

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_gmm_correlation(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's correlation using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    clusters = sklearn.mixture.GaussianMixture(
        n_components=3, covariance_type="full", n_init=10, random_state=random_state
    ).fit_predict(np.corrcoef(data, rowvar=False))

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_gmm_voltage(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's voltage using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    clusters = sklearn.mixture.GaussianMixture(
        n_components=3, covariance_type="full", n_init=10, random_state=random_state
    ).fit_predict(data.T)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_hierarchical_correlation( # pylint: disable=unused-argument
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's correlation using hierarchical clustering.

    Using precomputed correlation based pairwise distances perform worse on the dataset being used.
    Clustering the correlation matrix directly gives better results.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    clusters = sklearn.cluster.AgglomerativeClustering(
        n_clusters=3,
        affinity="euclidean",
        linkage="ward",
    ).fit_predict(np.corrcoef(data, rowvar=False))

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_hierarchical_voltage( # pylint: disable=unused-argument
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's voltage using hierarchical clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    clusters = sklearn.cluster.AgglomerativeClustering(
        n_clusters=3,
        affinity="euclidean",
        linkage="ward",
    ).fit_predict(data.T)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_kmeans_correlation(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's correlation using k-means clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    clusters = sklearn.cluster.KMeans(
        n_clusters=3, n_init=10, random_state=random_state
    ).fit_predict(np.corrcoef(data, rowvar=False))

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_kmeans_voltage(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's voltage using k-means clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    clusters = sklearn.cluster.KMeans(
        n_clusters=3, n_init=10, random_state=random_state
    ).fit_predict(data.T)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_kmedoids_correlation(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's correlation using k-medoids clustering.

    Using precomputed correlation based pairwise distances perform worse on the dataset being used.
    Clustering the correlation matrix directly gives better results.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    rng = np.random.default_rng(seed=random_state)
    run_count = 10

    best_clusters = None
    best_score = -np.inf

    for _ in range(run_count):
        random_state_cluster = rng.integers(np.iinfo(np.int32).max)

        clusters = sklearn_extra.cluster.KMedoids(
            n_clusters=3, method="pam", metric="euclidean", init="k-medoids++",
            random_state=random_state_cluster
        ).fit_predict(np.corrcoef(data, rowvar=False))

        score = sklearn.metrics.calinski_harabasz_score(X=data.T, labels=clusters)

        if score > best_score:
            best_score = score
            best_clusters = clusters

    return best_clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_kmedoids_voltage(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster load's voltage using k-medoids clustering.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    rng = np.random.default_rng(seed=random_state)
    run_count = 10

    best_clusters = None
    best_score = -np.inf

    for _ in range(run_count):
        random_state_cluster = rng.integers(np.iinfo(np.int32).max)

        clusters = sklearn_extra.cluster.KMedoids(
            n_clusters=3, method="pam", metric="euclidean", init="k-medoids++",
            random_state=random_state_cluster
        ).fit_predict(data.T)

        score = sklearn.metrics.calinski_harabasz_score(X=data.T, labels=clusters)

        if score > best_score:
            best_score = score
            best_clusters = clusters

    return best_clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_spectral_correlation(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster loads using spectral clustering with the one-plus-correlation matrix as the weighted
    adjacency matrix.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    # Note: The implementation of `SpectralClustering` in the version of sklearn used, at the time
    # of this writing, for the given dataset, will not converge to a solution when using `rbf`. To
    # get around this, we split the process into two steps. `SpectralClustering` will include the
    # first eigenvector (whose components are approximately the same constant), whereas
    # `SpectralEmbedding` will, by default, exclude this first eigenvector (the first eigenvalue,
    # from analysis, is close to zero).
    #
    # The choice of 2 components is based on analysis of the eigenvalues from a window of voltage
    # data.
    data_se = sklearn.manifold.SpectralEmbedding(
        n_components=2, affinity="precomputed"
    ).fit_transform(1 + np.corrcoef(data, rowvar=False))

    clusters = sklearn.cluster.KMeans(
        n_clusters=3, n_init=10, random_state=random_state
    ).fit_predict(data_se)

    return clusters

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_spectral_voltage(
    data: NDArray[(Any, Any), float], random_state: int
) -> NDArray[(Any, Any), int]:
    """
    Cluster loads using spectral clustering using the Rooted Normalized One-Minus-Correlation as a
    pairwise distance.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The voltage magnitude data.
    random_state : int
        The random state to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted clusters.
    """
    # Note: The implementation of `SpectralClustering` in the version of sklearn used, at the time
    # of this writing, for the given dataset, will not converge to a solution when using `rbf`. To
    # get around this, we split the process into two steps. `SpectralClustering` will include the
    # first eigenvector (whose components are approximately the same constant), whereas
    # `SpectralEmbedding` will, by default, exclude this first eigenvector (the first eigenvalue,
    # from analysis, is close to zero).
    #
    # The choice of 2 components is based on analysis of the eigenvalues from a window of voltage
    # data.
    data_se = sklearn.manifold.SpectralEmbedding(
        n_components=2, affinity="rbf"
    ).fit_transform(data.T)

    clusters = sklearn.cluster.KMeans(
        n_clusters=3, n_init=10, random_state=random_state
    ).fit_predict(data_se)

    return clusters
