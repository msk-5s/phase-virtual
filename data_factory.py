# SPDX-License-Identifier: MIT

"""
This module contains a factory for making data and labels.
"""

from typing import Any, Dict
from nptyping import NDArray

import pyarrow.feather
import numpy as np
import scipy

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data() -> NDArray[(Any, Any), float]:
    """
    Make the load data.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    """
    data = pyarrow.feather.read_feather("data/load_voltage.feather").to_numpy(dtype=float)

    return data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_gauss_noise(
    data: NDArray[(Any, Any), float], percent: float, random_state: int
) -> NDArray[(Any, Any), float]:
    """
    Make the load data with additive Gaussian noise.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    percent : float
        The percentage of Gaussian noise to add.
    random_state : int
        The state to use for rng.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The data with additive Gaussian noise.
    """
    # ~99.7% probability of noisy value being within `percent_gauss` of the true value.
    gauss = scipy.stats.norm.rvs(
        loc=0, scale=(data * percent) / 3, size=data.shape, random_state=random_state
    )

    return data + gauss

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_laplace_noise(
    data: NDArray[(Any, Any), float], percent: float, random_state: int
) -> NDArray[(Any, Any), float]:
    """
    Make the load data with additive Laplacian noise.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    percent : float
        The percentage of Laplacian noise to add.
    random_state : int
        The state to use for rng.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The data with additive Laplacian noise.
    """
    laplace = scipy.stats.laplace.rvs(
        loc=0, scale=(data * percent) / 3, size=data.shape, random_state=random_state
    )

    return data + laplace

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_phase_labels(percent_error: float, random_state: int) -> Dict[str, NDArray[(Any,), int]]:
    """
    Make the true and erroneous phase labels.

    Parameters
    ----------
    percent_error : float
        The percentage of phase labels to make incorrect.
    random_state: int
        The random state to use for rng.

    Returns
    -------
    dict of [str, NDArray[(Any,), int]]
        error : numpy.ndarray of int, (n_load,)
            The erroneous phase labels.
        true : numpy.ndarray of int, (n_load,)
            The true phase labels.
    """
    labels_true = pyarrow.feather.read_feather("data/metadata.feather")["phase"].to_numpy(dtype=int)

    rng = np.random.default_rng(random_state)

    label_count = len(labels_true)
    error_count = int(label_count * percent_error)
    indices = rng.permutation(label_count)[:error_count]

    unique_count = len(np.unique(labels_true))
    labels_error = labels_true.copy()

    # Increment the original label by 1 and wrap around when appropriate.
    labels_error[indices] = (labels_error[indices] + 1) % unique_count

    labels = {
        "error": labels_error,
        "true": labels_true
    }

    return labels

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_xfmr_labels() -> NDArray[(Any,), int]:
    """
    Make the true labels for the transformers.

    The transformer labels indicate which transformer a load is connected to.

    Returns
    -------
    xfmr_labels : numpy.ndarray of int, (n_load,)
        The transformer labels for each load.
    """
    xfmr_names = pyarrow.feather.read_feather(
        "data/metadata.feather"
    )["transformer_name"].to_numpy(dtype=str)

    # Convert the transformer names to integers.
    (_, xfmr_labels) = np.unique(xfmr_names, return_inverse=True)

    return xfmr_labels
