# SPDX-License-Identifier: MIT

"""
This script plots the output of spectral embedding.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather
import seaborn as sns
import scipy.sparse.csgraph
import sklearn.manifold

import convert
import data_factory
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals,too-many-statements
    """
    The main function.
    """
    # We want the randomness to be repeatable.
    rng = np.random.default_rng(seed=1337)
    random_state = rng.integers(np.iinfo(np.int32).max)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    xfmr_labels = data_factory.make_xfmr_labels()

    start = 0
    width = 96 * 7

    data = data_factory.make_data()[start:(start + width), :]
    data_n = data_factory.make_data_gauss_noise(data=data, percent=0.005, random_state=random_state)
    data_vn = convert.load_to_virtual(data=data_n, xfmr_labels=xfmr_labels)

    #***********************************************************************************************
    # Create phase labels.
    #***********************************************************************************************
    phase_labels = data_factory.make_phase_labels(
        percent_error=0.4, random_state=random_state
    )["true"]

    xfmr_names = pyarrow.feather.read_feather(
        "data/metadata.feather"
    )["transformer_name"].to_numpy(dtype=str)

    # Convert the transformer names to integers.
    (_, xfmr_labels) = np.unique(xfmr_names, return_inverse=True)

    xfmr_count = len(np.unique(xfmr_labels))
    xfmr_indices = [np.where(xfmr_labels == i)[0] for i in range(xfmr_count)]

    # Loads connected to the same transformer will have the same phase. We can just use the phase
    # label of the first load in `indices`.
    phase_labels_v = np.array([phase_labels[indices[0]] for indices in xfmr_indices])

    #***********************************************************************************************
    # Sort data and labels by phase. This is for visualization purposes.
    #***********************************************************************************************
    sort_indices = np.argsort(phase_labels)
    sort_indices_v = np.argsort(phase_labels_v)

    phase_labels = phase_labels[sort_indices]
    phase_labels_v = phase_labels_v[sort_indices_v]

    data_n = data_n[:, sort_indices]
    data_vn = data_vn[:, sort_indices_v]

    #***********************************************************************************************
    # Make the data stationary.
    #***********************************************************************************************
    order = 10
    cutoff = 0.06

    data_ns = transform.filter_butterworth(data=data_n, cutoff=cutoff, order=order)
    data_vns = transform.filter_butterworth(data=data_vn, cutoff=cutoff, order=order)

    #***********************************************************************************************
    # Dimensionality reduction via spectral embedding.
    #***********************************************************************************************
    se_n = sklearn.manifold.SpectralEmbedding(
        n_components=8, affinity="rbf", random_state=random_state
    ).fit(data_ns.T)

    se_vn = sklearn.manifold.SpectralEmbedding(
        n_components=8, affinity="rbf", random_state=random_state
    ).fit(data_vns.T)

    #***********************************************************************************************
    # Get eigenvalues and eigenvectors of the laplacian matrices.
    #***********************************************************************************************
    laplacian_n = scipy.sparse.csgraph.laplacian(se_n.affinity_matrix_, normed=True)
    laplacian_vn = scipy.sparse.csgraph.laplacian(se_vn.affinity_matrix_, normed=True)

    (values_n, vectors_n) = np.linalg.eig(laplacian_n)
    (values_vn, vectors_vn) = np.linalg.eig(laplacian_vn)

    indices_n = np.argsort(values_n)
    indices_vn = np.argsort(values_vn)

    values_n = values_n[indices_n]
    values_vn = values_vn[indices_vn]

    vectors_n = vectors_n[:, indices_n]
    vectors_vn = vectors_vn[:, indices_vn]

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    fontsize = 60
    legend_fontsize = 40
    plt.rc("axes", labelsize=fontsize)
    plt.rc("axes", titlesize=fontsize)
    plt.rc("legend", title_fontsize=legend_fontsize)
    plt.rc("legend", fontsize=legend_fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot eigenvalues and eigenvectors.
    #***********************************************************************************************
    frame_n = pd.DataFrame({
        "Eigenvector 2": vectors_n[:, 1],
        "Eigenvector 3": vectors_n[:, 2],
        "Phase": [["A", "B", "C"][x] for x in phase_labels]
    })

    frame_vn = pd.DataFrame({
        "Eigenvector 2": vectors_vn[:, 1],
        "Eigenvector 3": vectors_vn[:, 2],
        "Phase": [["A", "B", "C"][x] for x in phase_labels_v]
    })

    value_count = 30
    value_size = 300

    #(fig_values_n, axs_values_n) = plt.subplots()
    #fig_values_n.tight_layout()
    #axs_values_n.scatter(
    #    np.arange(2, value_count + 1), np.log(values_n[1:value_count]), s=value_size
    #)
    #axs_values_n.set_xticks(np.arange(2, value_count + 1, 2))
    #axs_values_n.set_xlabel("Eigenvalue")
    #axs_values_n.set_ylabel("log(Value)")

    (fig_values_vn, axs_values_vn) = plt.subplots()
    fig_values_vn.tight_layout()
    axs_values_vn.scatter(
        np.arange(2, value_count + 1), np.log(values_vn[1:value_count]), s=value_size
    )
    axs_values_vn.set_xticks(np.arange(2, value_count + 1, 2))
    axs_values_vn.set_xlabel("Eigenvalue")
    axs_values_vn.set_ylabel("log(Value)")

    scatter_size = 100

    (fig_vectors_n, axs_vectors_n) = plt.subplots()
    fig_vectors_n.tight_layout()
    sns.scatterplot(
        data=frame_n, x="Eigenvector 2", y="Eigenvector 3", hue="Phase", ax=axs_vectors_n,
        s=scatter_size
    )
    axs_vectors_n.legend(markerscale=3)

    (fig_vectors_vn, axs_vectors_vn) = plt.subplots()
    fig_vectors_vn.tight_layout()
    sns.scatterplot(
        data=frame_vn, x="Eigenvector 2", y="Eigenvector 3", hue="Phase", ax=axs_vectors_vn,
        s=scatter_size
    )
    axs_vectors_vn.legend(markerscale=3)

    #fig_values_n.suptitle("Eigenvalues: Load")
    #fig_values_vn.suptitle("Eigenvalues: Virtual")
    #fig_vectors_n.suptitle("Eigenvectors: Load")
    #fig_vectors_vn.suptitle("Eigenvectors: Virtual")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
