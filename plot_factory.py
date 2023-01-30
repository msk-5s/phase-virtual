# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains a factory for making plots.
"""

from typing import Any, Optional, Tuple
from nptyping import NDArray

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_correlation_heatmap(
    data: NDArray[(Any, Any), float], labels: NDArray[(Any,), int], aspect: str = "equal"
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a heatmap of the correlations between loads.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_load)
        The load data to plot.
    labels : numpy.ndarray, (n_load,)
        The phase labels of the loads.
    aspect : str, default="equal", ["auto", "equal"]
        The aspect mode to use.
    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    # Sort the loads by phase so we can better see how the phases follow the correlation structure
    # of the voltage measurements.
    sort_indices = np.argsort(labels)

    #***********************************************************************************************
    # Create the dotted boundaries to seperated the loads by phase more distinctly.
    # We want to place the phase labels inbetween the boundaries.
    #***********************************************************************************************
    # Boundaries to show the phases more distinctly using horizontal and vertical lines.
    phase_counts = np.bincount(labels)
    boundaries = [phase_counts[0], phase_counts[0] + phase_counts[1]]

    # Tick positions for the phases in the graph axis. This allows us to position the labels "A",
    # "B", etc. inbetween each boundary lines.
    tick_positions = [
        boundaries[0] // 2,
        boundaries[0] + ((boundaries[1] - boundaries[0]) // 2),
        boundaries[1] + ((len(labels) - boundaries[1]) // 2)
    ]

    tick_labels = ["A", "B", "C"]

    #***********************************************************************************************
    # Make the heatmap and add a color bar on the side to show the correlation value/color
    # relationship.
    #***********************************************************************************************
    (figure, axs) = plt.subplots()

    cor = np.corrcoef(data[:, sort_indices], rowvar=False)

    image = axs.imshow(cor, aspect=aspect)

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap=image.cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("Coefficient")

    #***********************************************************************************************
    # Label the plot's axis' and plot the boundary lines.
    #***********************************************************************************************
    figure.tight_layout()
    axs.set_xlabel("Load")
    axs.set_ylabel("Load")
    axs.set_xticks(tick_positions)
    axs.set_yticks(tick_positions)
    axs.set_xticklabels(tick_labels)
    axs.set_yticklabels(tick_labels)

    for boundary in boundaries:
        axs.axhline(y=boundary, color="red", linestyle="dashed", linewidth=5)
        axs.axvline(x=boundary, color="red", linestyle="dashed", linewidth=5)

    return (figure, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_periodogram_filter_plot(
    filter_dft: NDArray[(Any,), complex], frequencies: NDArray[(Any,), float]
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a periodogram plot of a filter.

    Parameters
    ----------
    filter_dft : numpy.ndarray of complex, (n_timestep,)
        The filter in the frequency domain.
    frequencies : numpy.ndarray of float, (n_timestep,)
        The the values for the frequency axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    magnitudes = np.abs(filter_dft)

    (figure, axs) = plt.subplots()

    figure.tight_layout()
    axs.plot(np.fft.fftshift(frequencies), np.fft.fftshift(magnitudes))

    axs.set_xlabel(r"$\omega$ (Cycles/Sample)")
    axs.set_ylabel(r"$|H(\omega)|$")

    return (figure, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_periodogram_voltage_plot(
    series_dft: NDArray[(Any,), complex],
    frequencies: NDArray[(Any,), float],
    cutoff: Optional[float] = None
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a periodogram plot of a voltage time series.

    Parameters
    ----------
    series_dft : numpy.ndarray of complex, (n_timestep,)
        The time series in the frequency domain.
    frequencies : numpy.ndarray of float, (n_timestep,)
        The the values for the frequency axis.
    cutoff : optional of float, default=None
        The cutoff frequency. A dashed vertical line will be drawn at the cutoff frequency.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    # We use the decibel scale since the DC component can make it difficult to visualize the higher
    # frequencies.
    magnitudes = (10 * np.log10(np.abs(series_dft)))

    (figure, axs) = plt.subplots()

    figure.tight_layout()
    axs.plot(np.fft.fftshift(frequencies), np.fft.fftshift(magnitudes))

    if cutoff is not None:
        axs.axvline(x=cutoff, color="r", linestyle="dashed", linewidth=5)
        axs.axvline(x=-float(cutoff), color="r", linestyle="dashed", linewidth=5)

    axs.set_xlabel(r"$\omega$ (Cycles/Sample)")
    axs.set_ylabel("(dBV)")

    return (figure, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_voltage_series_plot(
    series: NDArray[(Any,), float]
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a plot of a voltage time series.

    Parameters
    ----------
    series : numpy.ndarray of float, (n_timestep,)
        The time series to plot.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    (figure, axs) = plt.subplots()

    figure.tight_layout()
    axs.plot(series)
    axs.set_xlabel("Time (15-min)")
    axs.set_ylabel(r"$v(t)$ (V)")

    return (figure, axs)
