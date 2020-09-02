"""
## Visualisation utilities
--------------------------------------------------
## Author: Callum J. Court.
## Email: cc889@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Callum Court & Batuhan Yildirim 2020, ICSG3D
-------------------------------------------------
"""


import os
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
from sklearn.manifold import TSNE

import cv2


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def viz(
    sample, name="plot.png", show=True, alpha=0.2, ax=None, resample_d=(20, 20, 20)
):
    sample = resize(sample, resample_d)  # resize, otherwise it's super slow
    colours = cm.viridis(sample)
    colours = explode(colours)
    filled = colours[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    ax.voxels(x, y, z, filled, facecolors=colours, alpha=alpha)
    if show:
        plt.show(block=True)
        plt.close()
    else:
        return ax


def viz_duo(x_sample, y_sample, name="test.png", show=True, alpha=0.2):
    x_sample = resize(x_sample, (12, 12, 12))  # resize, otherwise it's super slow
    y_sample = resize(y_sample, (12, 12, 12))  # resize, otherwise it's super slow

    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Real")
    colours = cm.viridis(x_sample)
    colours = explode(colours)
    filled = colours[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x, y, z, filled, facecolors=colours, alpha=alpha)

    ax = fig.add_subplot(122, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Predicted")

    colours = cm.viridis(y_sample)
    colours = explode(colours)
    filled = colours[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x, y, z, filled, facecolors=colours, alpha=alpha)

    if show:
        plt.show()
    plt.close()


def animate(m, s, d=32):
    print(m.shape, s.shape)
    plt.ion()
    fig, axes = plt.subplots(1, 2)
    for i in range(d):
        axes[0].imshow(m[:, :, i])
        axes[1].imshow(s[:, :, i])
        plt.pause(0.2)
        axes[0].clear()
        axes[1].clear()
    plt.close()
    return


def animate_numeric(m, s, d=32):
    print(m.shape, s.shape)
    plt.ion()
    fig, axes = plt.subplots(1, 2)
    for i in range(d):
        axes[0].clear()
        axes[1].clear()

        for ix in range(d):
            for jx in range(d):
                tm = float("%.1f" % m[ix, jx, i])
                ts = float("%.1f" % s[ix, jx, i])
                if ts != 0:
                    textm = axes[0].text(
                        ix, jx, tm, ha="center", va="center", color="b", fontsize=8
                    )
                    texts = axes[1].text(
                        ix, jx, ts, ha="center", va="center", color="b", fontsize=8
                    )

        axes[0].set_xlim(0, d)
        axes[1].set_xlim(0, d)
        axes[0].set_ylim(0, d)
        axes[1].set_ylim(0, d)

        input()
    plt.close()
    return


def viz_slice(x, d):
    fig, axes = plt.subplots(1, 1)
    plt.cla()
    axes.imshow(x[:, :, d])
    plt.show()
    plt.close()
    return


def imscatter(x, y, ax, imageData, zoom=1.0, frame=True):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]
        # img = img.astype(np.uint8)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=1.0)
        ab = AnnotationBbox(image, (x0, y0), xycoords="data", frameon=frame)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def tsne_latent(Zs, Ms):
    embedded = TSNE(n_components=2).fit_transform(Zs)
    fig, ax = plt.subplots()
    imscatter(embedded[:, 0], embedded[:, 1], ax, Ms, zoom=1.0)
    plt.show(block=True)
    plt.close()
    return


def points(S, ax=None, show=True, d=32):
    if len(S.shape) > 2:
        S = S.reshape(d, d, d, S.shape[-1])
        S = S[S[:, :, :, 0] != 0]

    elements = np.unique(S[:, 0])
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for elem in elements:
        xyz = S[S[:, 0] == elem]
        ax.scatter(xyz[:, -3], xyz[:, -2], xyz[:, -1], label=elem)

    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    ax.legend()
    if show:
        plt.show(block=True)
        plt.close()
    return ax


def plot_points_3d(S, alpha=0.5, ignore=[], ax=None):
    """ 3D scatter """
    a, b, c = S.shape
    xc = np.linspace(0, S.shape[0], S.shape[0])
    yc = np.linspace(0, S.shape[1], S.shape[1])
    zc = np.linspace(0, S.shape[2], S.shape[2])
    coords = np.array(list(product(xc, yc, zc))).reshape(a, b, c, 3)
    S = S.reshape(a, b, c, 1)
    unique = np.unique(S)
    scoords = np.concatenate([S, coords], axis=-1)
    scoords = scoords[scoords[:, :, :, 0] != 0]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for un in unique:
        if un in ignore:
            continue
        mask = scoords[:, 0] == un
        smask = scoords[mask]
        ax.scatter(smask[:, 1], smask[:, 2], smask[:, 3], label=un, alpha=alpha)
        ax.set_xlim(0, a)
        ax.set_ylim(0, b)
        ax.set_zlim(0, c)
    ax.legend()
    if ax is None:
        plt.show(block=True)
    return ax
