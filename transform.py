# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for performing transformations on data.
"""

from typing import Any
from nptyping import NDArray

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def filter_butterworth(
    data: NDArray[(Any, Any), float], cutoff: float, order: int
) -> NDArray[(Any, Any), float]:
    """
    Apply a butterworth high pass filter to each column (time series) of `data` to filter out the
    low frequency trend and seasonality.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to transform.
    cutoff : float
        The cutoff frequency in cycles per sample.
    order : int
        The order of the butterworth filter.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The transformed time series data.
    """
    # We add a small number to the frequencies to prevent division by zero.
    frequencies = np.fft.fftfreq(data.shape[0]) + 1e-6

    hpf = (1 / np.sqrt(1 + (cutoff / frequencies)**(2 * order))).astype(complex)

    filtered_data = np.zeros(shape=data.shape)

    for (i, series) in enumerate(data.T):
        series_dft = np.fft.fft(series)
        filtered_series_dft = series_dft * hpf

        filtered_data[:, i] = np.fft.ifft(filtered_series_dft).real

    return filtered_data
