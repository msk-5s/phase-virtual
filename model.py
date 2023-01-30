# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for performing phase label correction.
"""

from typing import Any, Callable, Optional
from nptyping import NDArray

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def predict( # pylint: disable=too-many-arguments
    data: NDArray[(Any, Any), float], phase_labels: NDArray[(Any,), int],
    predict_clusters: Callable[[NDArray[(Any, Any), float], int], NDArray[(Any, Any), int]],
    random_state: int, xfmr_labels: Optional[NDArray[(Any,), int]] = None
) -> NDArray[(Any, Any), int]:
    """
    Predict the phase labels using clustering.

    Parameters
    ----------
    data : numpy.ndarray of int, (n_timestep, n_load)
        The voltage magnitude data.
    phase_labels : numpy.ndarray of int, (n_load,)
        The phase labels.
    predict_clusters: callable
        The clustering function to use. See `clusterer.py`.
    random_state : int
        The random state to use.
    xfmr_labels : optional of numpy.ndarray of int, (n_load,), default=None
        The transformer labels.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted phase labels.
    """
    clusters = predict_clusters(data=data, random_state=random_state)

    if xfmr_labels is not None:
        clusters = _virtual_clusters_to_load_clusters(v_clusters=clusters, xfmr_labels=xfmr_labels)

    predictions = _predict_majority_vote(clusters=clusters, labels=phase_labels)

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _predict_majority_vote(
    clusters: NDArray[(Any,), int], labels: NDArray[(Any,), int]
) -> NDArray[(Any,), int]:
    """
    Do label correction using predicted clusters and labels with a majority vote rule.

    Parameters
    ----------
    clusters : numpy.ndarray of int, (n_meter,)
        The predicted clusters for each meter.
    labels : numpy.ndarray of int, (n_meter,)
        The labels to use for the meters. These labels will be used in the majority vote approach.

    Returns
    -------
    numpy.ndarray of int, (n_meter,)
        The label predictions via majority vote.
    """
    unique_clusters = np.unique(clusters)

    indices_list = [np.where(clusters == i)[0] for i in unique_clusters]

    predictions = np.zeros(shape=len(clusters), dtype=int)

    for indices in indices_list:
        observed_labels = labels[indices]
        predicted_label = np.bincount(observed_labels).argmax()

        predictions[indices] = predicted_label

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _virtual_clusters_to_load_clusters(
    v_clusters: NDArray[(Any,), int], xfmr_labels: NDArray[(Any,), int]
) -> NDArray[(Any,), int]:
    """
    Assign the cluster labels of each virtual transformer to their connected loads.

    Parameters
    ----------
    v_clusters : numpy.ndarray of int, (n_transformer,)
        The cluster labels for the virtual transformers.
    xfmr_labels : numpy.ndarray of int, (n_load,)
        The load transformer labels.

    Returns
    -------
    load_clusters : numpy.ndarray of int, (n_load,)
        The cluster labels for the loads as determined by the virtual transformer clusters.
    """
    # Get a list of indices of loads connected to the same transformer.
    xfmr_count = len(np.unique(xfmr_labels))
    xfmr_indices = [np.where(xfmr_labels == i)[0] for i in range(xfmr_count)]

    load_clusters = np.zeros(shape=len(xfmr_labels), dtype=int)

    # Assign each virtual transformer cluster to the loads connected to that transformer.
    for (i, indices) in enumerate(xfmr_indices):
        load_clusters[indices] = v_clusters[i]

    return load_clusters
