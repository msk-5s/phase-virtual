# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for performing convertions on data.
"""

from typing import Any, Tuple
from nptyping import NDArray

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def load_to_virtual(
    data: NDArray[(Any, Any), float], xfmr_labels: NDArray[(Any,), int]
) -> Tuple[NDArray[(Any, Any), float]]:
    """
    Convert the load measurement data to virtual transformer measurements by averaging.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_load)
        The load measurement data.
    xfmr_labels : numpy.ndarray of int, (n_load,)
        The load transformer labels.

    Returns
    -------
    data_v : numpy.ndarray of float, (n_timestep, n_transformer)
        The virtual transformer measurements.
    """
    xfmr_count = len(np.unique(xfmr_labels))
    xfmr_indices = [np.where(xfmr_labels == i)[0] for i in range(xfmr_count)]

    data_v = np.zeros(shape=(data.shape[0], xfmr_count), dtype=float)

    for (i, indices) in enumerate(xfmr_indices):
        data_v[:, i] = np.mean(data[:, indices], axis=1)

    return data_v
