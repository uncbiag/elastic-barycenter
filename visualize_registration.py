import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_registration(source, target, registered):
    """
    Plots the registration from source to target
    Arguments:
        source (ndarray, torch.tensor)
        target (ndarray, torch.tensor)
        registered (ndarray, torch.tensor)
    Returns if the formats are not correct
    """
    # Convert everything to numpy
    if isinstance(source, np.ndarray):
        source = source / source.max()
    elif torch.is_tensor(source):
        source = source.detach().cpu().squeeze().numpy()
        source = source / source.max()
    else:
        return

    if isinstance(target, np.ndarray):
        target = target / target.max()
    elif torch.is_tensor(target):
        target = target.detach().cpu().squeeze().numpy()
        target = target / target.max()
    else:
        return

    if isinstance(registered, np.ndarray):
        registered = registered / registered.max()
    elif torch.is_tensor(registered):
        registered = registered.detach().cpu().squeeze().numpy()
        registered = registered / registered.max()
    else:
        return

    # Start plotting
    fontsize = 10
    fig, axs = plt.subplots(2, 3)
    plt.axis("off")

    # Plot the source
    a1 = axs[0, 0].imshow(source, cmap="gray")
    axs[0, 0].set_title("Source", fontsize=fontsize)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a1, cax=cax)

    # Plot the registered image
    a2 = axs[0, 1].imshow(registered, cmap="gray")
    axs[0, 1].set_title("Registered", fontsize=fontsize)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a2, cax=cax)

    # Plot the target image
    a3 = axs[0, 2].imshow(target, cmap="gray")
    axs[0, 2].set_title("Target", fontsize=fontsize)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a3, cax=cax)

    # Plot the difference between the target and the registered source
    a4 = axs[1, 0].imshow(target - registered, vmin=-0.5, vmax=0.5, cmap="gray")
    axs[1, 0].set_title("t - r", fontsize=fontsize)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a4, cax=cax)

    # Plot the difference between the target and the source
    a5 = axs[1, 1].imshow(target - source, vmin=-0.5, vmax=0.5, cmap="gray")
    axs[1, 1].set_title("t - s", fontsize=fontsize)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a5, cax=cax)

    plt.show()


def plot_deformation_grid(phi, ax=None):
    """
    Takes a registration function phi and plots a deformation grid
    Arguments:
        phi (torch.tensor): the registration function
    """
    x_d = phi[:, :, 0].detach().cpu().numpy()
    y_d = phi[:, :, 1].detach().cpu().numpy()

    if ax is None:
        contourx = plt.contour(x_d, levels=np.linspace(x_d.min(), x_d.max(), 30), colors="red", linewidths=0.5, linestyles="solid")
        contoury = plt.contour(y_d, levels=np.linspace(y_d.min(), y_d.max(), 30), colors="red", linewidths=0.5, linestyles="solid")
        plt.title("Deformation Grid")
        ax1 = contourx.axes
        ax2 = contoury.axes
        ax1.set_aspect('equal', 'box')
        ax2.set_aspect('equal', 'box')
        ax2.invert_yaxis()
        plt.show()
    else:
        ax.contour(x_d, levels=np.linspace(x_d.min(), x_d.max(), 30), colors="red", linewidths=0.5, linestyles="solid")
        ax.contour(y_d, levels=np.linspace(y_d.min(), y_d.max(), 30), colors="red", linewidths=0.5, linestyles="solid")
        ax.set_aspect('equal', 'box')
        ax.set_title("Deformation Grid")
        ax.invert_yaxis()
        return ax


def plot_image_deformation(registered, phi, ax=None):
    """
    Takes a registered image and its registration function phi and plots the image with a superimposed registration grid
    Arguments:
        registered (torch.tensor): the registered image
        phi (torch.tensor): the registration fun
    """
    if ax is not None:
        ax.imshow(registered.detach().cpu().numpy(), vmin=0, cmap="gray")
        ax = plot_deformation_grid(phi, ax)
        return ax
    else:
        plt.imshow(registered.detach().cpu().numpy(), vmin=0, cmap="gray")
        plot_deformation_grid(phi)