import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def barycenter(imagestack, outer_epochs, inner_epochs, lr, gamma, verbose=False):
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # Creating the grid
    h, w, l = imagestack.shape
    y = torch.arange(0, h)
    x = torch.arange(0, w)
    meshy, meshx = torch.meshgrid((y, x))
    meshy = 2 * (meshy / h) - 1
    meshx = 2 * (meshx / w) - 1
    grid = torch.stack((meshx, meshy), 2)
    x = grid.clone().to(device).type(dtype)

    # Converting arrays to torch tensors
    image_tensors = []
    for i in range(l):
        normalized_image = imagestack[:, :, i] / np.amax(imagestack[:, :, i])
        image_tensors.append(torch.tensor(normalized_image, dtype=dtype, device=device))

    # Parameter tensors
    params = []
    for i in range(len(image_tensors)):
        params.append(torch.zeros_like(x, dtype=dtype, device=device, requires_grad=True))

    # Some auxiliary variables for the loss
    dx1 = x[0, 1, 0] - x[0, 0, 0]
    dx2 = x[1, 0, 1] - x[0, 0, 1]

    # Visualization
    if verbose:
        delta = 5 / (outer_epochs - 1)
        display_its = [int(t / delta) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]
        losses = [[] for t in imagestack]
        losses_d = [[] for t in imagestack]
        losses_r = [[] for t in imagestack]
        progress_images = []

    # Parameters for gradient descent
    transformations = image_tensors
    optimizer = torch.optim.Adam(params, lr=lr)

    # Outer loop that updates Barycenter
    for t in range(outer_epochs):
        # Forward pass
        optimizer.zero_grad()
        I_b = torch.zeros_like(image_tensors[0], device=device, dtype=dtype)
        for i in range(len(transformations)):
            I_b += transformations[i]
        I_b = 1 / l * I_b

        # Inner loop for registration
        transformations = []
        for i in range(len(params)):
            u = params[i]
            for t2 in range(inner_epochs):
                phi_i = x + u
                u_x1 = 1 / dx1 * torch.cat(
                    (u[:, 1:, 0] - u[:, :-1, 0], torch.zeros((u.shape[0], 1), device=device, dtype=dtype)), 1)
                u_x2 = 1 / dx2 * torch.cat(
                    (u[1:, :, 0] - u[:-1, :, 0], torch.zeros((1, u.shape[1]), device=device, dtype=dtype)), 0)
                u_y1 = 1 / dx1 * torch.cat(
                    (u[:, 1:, 1] - u[:, :-1, 1], torch.zeros((u.shape[0], 1), device=device, dtype=dtype)), 1)
                u_y2 = 1 / dx2 * torch.cat(
                    (u[1:, :, 1] - u[:-1, :, 1], torch.zeros((1, u.shape[1]), device=device, dtype=dtype)), 0)
                u_x = torch.stack((u_x1, u_x2), 2)
                u_y = torch.stack((u_y1, u_y2), 2)

                # Gridsample
                I_i = image_tensors[i]
                I_0_phi_i = F.grid_sample(I_i.view(1, 1, h, w), phi_i.unsqueeze(0), align_corners=True).squeeze()

                # Loss
                jacobian = (1 + u_x1) * (1 + u_y2) - u_x2 * u_y1
                loss_d = ((I_0_phi_i - I_b).pow(2) * dx1 * dx2).sum()
                loss_r = gamma * ((u_x.pow(2).sum(2) + u_y.pow(2).sum(2)) * dx1 * dx2).sum()
                loss = loss_d + loss_r
                loss.backward()
                if verbose:
                    losses[i].append(loss.item())
                    losses_d[i].append(loss_d.item())
                    losses_r[i].append(loss_r.item())

                # Backward pass
                optimizer.step()

            # Appending last transformation
            transformations.append(I_0_phi_i.detach())

            # Visualizing the last transformation
            if verbose:
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(I_i.detach().cpu().numpy(), cmap="gray")
                axs[0].set_title("Source")
                axs[1].imshow(I_0_phi_i.detach().cpu().numpy(), cmap="gray")
                axs[1].set_title("Registered")
                axs[2].imshow(I_b.detach().cpu().numpy(), cmap="gray")
                axs[2].set_title("Target")
                plt.show()

        # Plotting
        if t in display_its:
            I_b_ = (I_b).detach().cpu().numpy()
            progress_images.append(I_b_)

    if verbose:
        k = 1
        for t in range(len(display_its)):
            ax = plt.subplot(2, 3, k)
            k += 1
            ax.imshow(progress_images[t], vmin=0, vmax=1, cmap="gray")
            ax.set_title("t = {}".format(display_its[t]))
        plt.show()

        for i in range(len(losses)):
            plt.plot(range(outer_epochs * inner_epochs), losses[i], label="Total Loss")
            plt.plot(range(outer_epochs * inner_epochs), losses_d[i], label="Dissimilarity")
            plt.plot(range(outer_epochs * inner_epochs), losses_r[i], label="Regularizer")
            plt.title("Loss")
            plt.ylabel("SSD")
            plt.xlabel("epochs")
            plt.legend()
            plt.show()

    return I_b.detach()