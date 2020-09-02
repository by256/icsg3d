"""
## Functions for computing watershed segmentation
--------------------------------------------------
## Author: Callum J. Court.
## Email: cc889@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Callum Court & Batuhan Yildirim 2020, ICSG3D
-------------------------------------------------
"""


from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage import filters, measure, morphology

from viz import plot_points_3d


def get_background(S, kernel_size=1):
    kernel = morphology.ball(kernel_size)
    return morphology.dilation(S, kernel)


def get_foreground(S, kernel_size=1, erode=True):
    if not erode:
        return S
    else:
        kernel = morphology.ball(kernel_size)
        return morphology.erosion(S, kernel)


def crop(a, bbox):
    return a[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]]


def segment_nuclei(
    binary,
    species,
    intensity,
    wmin=8,
    it=1,
    max_iters=5,
    min_convexity=0.8,
    verbose=False,
):
    """ Computes segmented form of species matrix using recursive watershed segmentation """
    # Matrix for storing result
    R = np.zeros(binary.shape)
    binary = binary.astype(int)

    # 1. Label the connected components
    labels = measure.label(binary, connectivity=1)
    seg_classes, seg_counts = np.unique(labels, return_counts=True)
    seg_classes = np.array(
        [seg_classes[i] for i in range(len(seg_classes)) if seg_counts[i] > 3]
    )
    seg_classes = seg_classes[seg_classes != 0]
    if verbose:
        print("\nIteration", it)
        print("Classes", seg_classes)
        print("Counts", seg_counts)
        plot_points_3d(labels)

    for cl in seg_classes:
        if verbose:
            print("Class", cl)

        # Crop the images
        binary_cl = np.where(labels == cl, labels, 0)
        intensity_cl = np.where(labels == cl, intensity, 0)
        species_cl = np.where(labels == cl, species, 0)
        region = measure.regionprops(binary_cl, intensity_cl)
        bbox = region[0].bbox

        binary_bbox = crop(binary_cl, bbox)
        intensity_bbox = crop(intensity_cl, bbox)
        species_bbox = crop(species_cl, bbox)
        chull = morphology.convex_hull_image(binary_bbox)
        convexity = np.count_nonzero(binary_bbox) / np.count_nonzero(chull)

        if verbose:
            print("Convexity:", convexity)

        if convexity >= min_convexity:
            max_class = np.max(R)
            R[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]] = np.where(
                binary_bbox == cl,
                max_class + 1,
                R[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]],
            )
            continue

        # Get the foreground, bg etc.
        # Determine wether or not to erode
        fg = get_foreground(binary_bbox)
        bg = get_background(binary_bbox)
        unknown = bg - fg

        if verbose:
            print("Segmenting")
            plot_points_3d(fg)

        # Markers for ws
        markers = measure.label(fg)
        markers += 1
        markers[unknown == 1] = 0

        # WS
        wss = morphology.watershed(binary_bbox, markers)
        wss[wss == 1] = 0
        max_class = np.max(R)
        wss = wss + max_class  # sub region with classes relabelled
        wss[wss == max_class] = 0
        nclasses = len(np.unique(wss)) - 1

        if verbose:
            print("WS", it, np.unique(wss, return_counts=True))
            plot_points_3d(wss)
            print(int(np.count_nonzero(wss) / wmin), nclasses)

        # Determine wether or not to segment again on the basis of convexity and object counts
        if (
            int(np.count_nonzero(wss) / wmin) > len(np.unique(wss)) - 1
            and it < max_iters
        ):
            if verbose:
                print("Segmenting again")
            Rp = segment_nuclei(
                wss,
                species_bbox,
                intensity_bbox,
                it=it + 1,
                verbose=verbose,
                max_iters=max_iters,
                min_convexity=min_convexity,
            )
            max_class = np.max(R)
            Rp = Rp + max_class  # sub region with classes relabelled
            Rp[Rp == max_class] = 0
            R[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]] = np.where(
                Rp != 0, Rp, R[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]]
            )
        else:
            R[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]] = np.where(
                wss != 0,
                wss,
                R[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]],
            )

    if verbose:
        print(it, np.unique(R, return_counts=True))

    return R


def majority_vote(seg_img, R, cl):
    """ Majority vote of class cl in a region R in segmented image"""
    binary_label_map = np.where(R == cl, seg_img, 0).astype(int)
    if np.count_nonzero(binary_label_map) == 0:
        return 0

    unique, counts = np.unique(binary_label_map, return_counts=True)
    unique_counts = sorted(list(zip(unique, counts)), key=lambda x: x[1])
    unique_counts = [i for i in unique_counts if i[0] != 0]
    specie = unique_counts[-1][0]
    return specie


def centroids(seg_img, R):
    """ Determine centroid of a region R in segmented image """
    classes = np.unique(R)[1:]
    atoms = []
    means = []

    xc = np.linspace(0, R.shape[0], R.shape[0] + 1)[:-1]
    yc = np.linspace(0, R.shape[0], R.shape[0] + 1)[:-1]
    zc = np.linspace(0, R.shape[0], R.shape[0] + 1)[:-1]
    coords = np.array(list(product(xc, yc, zc))).reshape(32, 32, 32, 3)

    seg_img_coords = np.concatenate([seg_img.reshape(32, 32, 32, 1), coords], axis=-1)

    for cl in classes:
        cmask = R == cl
        smask = seg_img_coords[cmask]
        specie = majority_vote(seg_img, R, cl)
        if specie != 0:
            means.append(np.mean(smask[:, 1:], axis=0))
            atoms.append(specie)
    return atoms, means


def watershed_clustering(M, S, Sb, max_iters=5, return_ws=False, verbose=False):
    """Determine centroids and species of atoms in the density/species matrices
    Returns the atom z numbers and means in voxel coordinates"""
    M = M.squeeze()
    S = S.squeeze()
    Sb = Sb.squeeze()
    R = segment_nuclei(Sb, S, M, max_iters=max_iters, verbose=verbose)
    atoms, means = centroids(S, R)
    if return_ws:
        return np.array(atoms), np.array(means), R
    else:
        return np.array(atoms), np.array(means)
