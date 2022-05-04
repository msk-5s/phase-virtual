# SPDX-License-Identifier: MIT

"""
This script plots the correlation matrices of the virtual voltage measurements.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather

import convert
import data_factory
import plot_factory
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    start = 0
    width = 96 * 7

    data = data_factory.make_data()[start:(start + width), :]
    xfmr_labels = data_factory.make_xfmr_labels()

    data_v = convert.load_to_virtual(data=data, xfmr_labels=xfmr_labels)

    #***********************************************************************************************
    # Calculate the true phase labels for each virtual transformer.
    #***********************************************************************************************
    metadata = pyarrow.feather.read_feather("data/metadata.feather")

    phase_labels = metadata["phase"].to_numpy(dtype=int)
    xfmr_names = metadata["transformer_name"].to_numpy(dtype=str)

    # Convert the transformer names to integers.
    (_, xfmr_labels) = np.unique(xfmr_names, return_inverse=True)

    xfmr_count = len(np.unique(xfmr_labels))
    xfmr_indices = [np.where(xfmr_labels == i)[0] for i in range(xfmr_count)]

    # Loads connected to the same transformer will have the same phase. We can just use the phase
    # label of the first load in `indices`.
    labels_v = np.array([phase_labels[indices[0]] for indices in xfmr_indices])

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    order = 10
    cutoff = 0.06

    data_vs = transform.filter_butterworth(data=data_v, cutoff=cutoff, order=order)

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    #fontsize = 60
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("figure", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot correlation matrices.
    #***********************************************************************************************
    (fig_v, axs_v) = plot_factory.make_correlation_heatmap(
        data=data_v, labels=labels_v
    )

    (fig_vs, axs_vs) = plot_factory.make_correlation_heatmap(
        data=data_vs, labels=labels_v
    )

    fig_v.suptitle("Unfiltered Load")
    fig_vs.suptitle("Filtered Load")

    axs_v.set_xlabel("Virtual Transformer")
    axs_v.set_ylabel("Virtual Transformer")
    axs_vs.set_xlabel("Virtual Transformer")
    axs_vs.set_ylabel("Virtual Transformer")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
