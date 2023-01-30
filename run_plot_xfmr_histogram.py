# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the histogram of the number of loads connected to the secondary transformers.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import data_factory

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    #***********************************************************************************************
    # Load transformer labels and load counts.
    #***********************************************************************************************
    xfmr_labels = data_factory.make_xfmr_labels()
    load_counts = np.bincount(xfmr_labels)

    #***********************************************************************************************
    # Print count info.
    #***********************************************************************************************
    count_info = {
        count: len(load_counts[load_counts == count])
    for count in np.unique(load_counts)}

    print(count_info)

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    fontsize = 40
    plt.rc("axes", labelsize=fontsize)
    plt.rc("axes", titlesize=fontsize)
    plt.rc("figure", titlesize=fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot histogram.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()
    fig.tight_layout()

    sns.histplot(data=load_counts, bins=len(np.unique(load_counts)), discrete=True, ax=axs)

    axs.set_xlabel("Load Count")
    axs.set_ylabel("Transformer Count")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
