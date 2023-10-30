import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from visualize_registration import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float


def registration(source, target, delta_epochs, lr, gamma, verbose=False):
    """
    Gradient-descent based registration using the Adam optimizer and following an elastic loss
    Arguments:
        source (img arr): Image that will be registered
        target (img arr): Target for registration
        delta_epochs: delta value to compute the number of epochs. It is smaller than 1
        lr (float): learning rate for gradient descent
        gamma (float): regularization parameter for registration
        verbose: print losses, print registration progress
    Returns:
        Given registration function phi = y + u, it returns the displacement vector u
    """
    # Creating a grid
    h1, w1 = source.shape
    h2, w2 = target.shape
    y1 = torch.arange(0, h1)
    y2 = torch.arange(0, h2)
    x1 = torch.arange(0, w1)
    x2 = torch.arange(0, w2)
    meshy1, meshx1 = torch.meshgrid((y1, x2))
    meshy2, meshx2 = torch.meshgrid((y2, x2))

    # Scaling the grid to interval [-1, 1]
    meshy1 = 2 * (meshy1 / h1) - 1
    meshx1 = 2 * (meshx1 / w1) - 1
    meshy2 = 2 * (meshy2 / h2) - 1
    meshx2 = 2 * (meshx2 / w2) - 1
    grid1 = torch.stack((meshx1, meshy1), 2)
    grid2 = torch.stack((meshx2, meshy2), 2)

    # Parameters for optimization
    epochs = int(5 / delta_epochs) + 1
    if verbose:
        display_its = [int(t / delta_epochs) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]

    # Tensors to optimize
    x = grid1.clone().to(device).type(dtype)
    y = grid2.clone().to(device).type(dtype)
    I_0 = torch.tensor(source / source.max(), dtype=dtype, device=device)
    I_1 = torch.tensor(target / target.max(), dtype=dtype, device=device)

    # Defining vector of movement parameter
    u = torch.zeros_like(y, dtype=dtype, device=device, requires_grad=True)
    params = [u]

    # Variables needed for the loss function
    dy1 = y[0, 1, 0] - y[0, 0, 0]
    dy2 = y[1, 0, 1] - y[0, 0, 1]

    # Optimizer
    optimizer = torch.optim.Adam(params, lr=lr)

    # Saving loss arrays if we want to register with verbose command
    if verbose:
        losses = []
        losses_d = []
        losses_r = []
        k = 1

    # Starting the gradient descent registration
    for t in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        phi_i = y + u

        # Gradient with boundary conditions
        u_x1 = 1 / dy1 * torch.cat((u[:, 1:, 0] - u[:, :-1, 0], torch.zeros((u.shape[0], 1), device=device, dtype=dtype)), 1)
        u_x2 = 1 / dy2 * torch.cat((u[1:, :, 0] - u[:-1, :, 0], torch.zeros((1, u.shape[1]), device=device, dtype=dtype)), 0)
        u_y1 = 1 / dy1 * torch.cat((u[:, 1:, 1] - u[:, :-1, 1], torch.zeros((u.shape[0], 1), device=device, dtype=dtype)), 1)
        u_y2 = 1 / dy2 * torch.cat((u[1:, :, 1] - u[:-1, :, 1], torch.zeros((1, u.shape[1]), device=device, dtype=dtype)), 0)
        u_x = torch.stack((u_x1, u_x2), 2)
        u_y = torch.stack((u_y1, u_y2), 2)

        # Gridsample
        I_0_phi_i = F.grid_sample(I_0.view(1, 1, h1, w2), phi_i.unsqueeze(0), align_corners=True).squeeze()

        # Loss
        jacobian = (1 + u_x1) * (1 + u_y2) - u_x2 * u_y1
        loss_d = ((I_0_phi_i - I_1).pow(2) * dy1 * dy2).sum()
        loss_r = gamma * ((u_x.pow(2).sum(2) + u_y.pow(2).sum(2)) * dy1 * dy2).sum()
        loss = loss_d + loss_r

        if verbose:
            losses.append(loss.item())
            losses_d.append(loss_d.item())
            losses_r.append(loss_r.item())

            # Plotting
            if t in display_its:
                ax = plt.subplot(2, 3, k)
                k += 1
                I_0_phi_i_ = (I_0_phi_i).detach().cpu().numpy()
                ax.imshow(I_0_phi_i_, cmap="gray")
                ax.set_title("t = {}".format(t))

        if t % 100 == 0:
            print("Epoch {} loss: {}".format(t, loss.item()))

        # Backward pass
        loss.backward()
        optimizer.step()

    if verbose:
        # Showing the progress of the registration
        plt.tight_layout()
        plt.show()

        # Plotting the losses
        plt.plot(range(epochs), losses, label="Total loss")
        plt.plot(range(epochs), losses_d, label="Dissimilarity")
        plt.plot(range(epochs), losses_r, label="Regularization")
        plt.title("Loss")
        plt.ylabel("SSD")
        plt.xlabel("epochs")
        plt.legend()
        plt.show()

        # Plotting the overal registration
        visualize_registration(source, target, I_0_phi_i)

        # Plotting the deformation grid
        plot_deformation_grid(phi_i)

        # Quantifying folds in the grid
        zero = torch.tensor(0, device=device, dtype=dtype)
        negative_j = torch.where(jacobian < zero, jacobian, zero)
        print('Number of folds = {}'.format(torch.count_nonzero(negative_j).item()))

        # Quantifying SSD Loss
        print("SSD Before: {}".format((I_0 - I_1).pow(2).sum().item()))
        print("SSD After: {}".format((I_0_phi_i.detach() - I_1).pow(2).sum().item()))

    return u.detach().cpu().numpy()