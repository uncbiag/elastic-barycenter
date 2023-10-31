import torch
from image_functions import *
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from visualize_registration import *


def main(verbose=False):
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    dtype = torch.float

    # Example 1: Concentric circles
    # N = 250
    # imgA = circle(N, 90)
    # imgB = circle(N, 70)
    # imgC = circle(N, 50)
    # imgD = circle(N, 30)

    # Example 2: Equal sized circles shifted along the diagonals
    N = 250
    imgA = circle(N, 70, (100, 100))
    imgB = circle(N, 70, (100, 150))
    imgC = circle(N, 70, (150, 100))
    imgD = circle(N, 70, (150, 150))

    # Example 3: Creating a cross inside a square
    # N = 250
    # imgA = square(N, 125, 0)
    # imgB = cross(N, 125, 0, 4)

    # Example 4: Reading images from a directory
    # N = 64
    # imgA = 1 - image_from_file("Toy/A.png")
    # imgB = 1 - image_from_file("Toy/B.png")
    # imgC = 1 - image_from_file("Toy/C.png")
    # imgD = 1 - image_from_file("Toy/D.png")

    # Path to store barycenter
    outpath = "Toy/barycenter.npy"

    # Creating the [-1, 1] grid for registration
    y, x = torch.meshgrid([torch.arange(0, N).to(device).type(dtype) / N] * 2)
    y = 2 * y - 1
    x = 2 * x - 1
    grid = torch.stack((x, y), 2)

    # Fixed tensors - NOTE: Comment I_C and I_D for Example 3
    xG = grid.clone().to(device).type(dtype)
    I_A = torch.tensor(imgA / imgA.max(), dtype=dtype, device=device)
    I_B = torch.tensor(imgB / imgB.max(), dtype=dtype, device=device)
    I_C = torch.tensor(imgC / imgC.max(), dtype=dtype, device=device)
    I_D = torch.tensor(imgD / imgD.max(), dtype=dtype, device=device)

    # Blurring the tensors to make the computation of the barycenter easier
    # NOTE: Comment I_C and I_D for Example 3
    kernel = 11  # Must be an odd number
    I_A = T.functional.gaussian_blur(I_A.unsqueeze(0), kernel_size=kernel).squeeze()
    I_B = T.functional.gaussian_blur(I_B.unsqueeze(0), kernel_size=kernel).squeeze()
    I_C = T.functional.gaussian_blur(I_C.unsqueeze(0), kernel_size=kernel).squeeze()
    I_D = T.functional.gaussian_blur(I_D.unsqueeze(0), kernel_size=kernel).squeeze()

    # Visualizing individual images
    if verbose:
        # NOTE: Needs some easy modifications for Example 3
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(I_A.detach().cpu().numpy(), vmax=1, cmap="gray")
        axs[0].set_title("img1")
        axs[1].imshow(I_B.detach().cpu().numpy(), vmax=1, cmap="gray")
        axs[1].set_title("img2")
        axs[2].imshow(I_C.detach().cpu().numpy(), vmax=1, cmap="gray")
        axs[2].set_title("img3")
        axs[3].imshow(I_D.detach().cpu().numpy(), vmax=1, cmap="gray")
        axs[3].set_title("img4")
        plt.show()

    # Create tensors needed for the computation of the barycenter
    dx1 = xG[0, 1, 0] - xG[0, 0, 0]  # Incremental 1
    dx2 = xG[1, 0, 1] - xG[0, 0, 1]  # Incremental 2
    fixed = [I_A, I_B, I_C, I_D]  # Fixed tensors for which their barycenter will be computed

    # Tensors to optimize - displacement vectors in phi_i = y_i + u_i
    # NOTE: Comment uC and uD for Example 3
    uA = torch.zeros_like(xG, dtype=dtype, device=device, requires_grad=True)
    uB = torch.zeros_like(xG, dtype=dtype, device=device, requires_grad=True)
    uC = torch.zeros_like(xG, dtype=dtype, device=device, requires_grad=True)
    uD = torch.zeros_like(xG, dtype=dtype, device=device, requires_grad=True)

    # Parameters for gradient descent
    delta = 0.5
    epochs = int(5 / delta) + 1
    inner_epochs = 1000
    lr = 0.001
    display_its = [int(t / delta) for t in [0, 0.5, 1.0, 2.0, 3.0, 5.0]]
    gamma = 0.1
    params = [uA, uB, uC, uD]
    phis = [xG, xG, xG, xG]
    transformations = fixed
    optimizer = torch.optim.Adam(params, lr=lr)

    # Lists needed for visualization of losses and progress of registration
    # NOTE: Create only a list of two lists for Example 3
    losses = [[], [], [], []]
    losses_d = [[], [], [], []]
    losses_r = [[], [], [], []]
    folds = [[], [], [], []]
    progress_images = []
    progress_images_grid = [[], [], [], []]
    progress_images_reg = [[], [], [], []]
    if verbose:
        delta_inner = 5 / (inner_epochs - 1)
        display_its_inner = [int(t / delta_inner) for t in [0, 0.5, 1.0, 2.0, 3.0, 5.0]]

    for t in range(epochs):

        # Updating barycenter
        I_b = torch.zeros_like(I_A, device=device, dtype=dtype)  # Resetting barycenter

        for I in range(len(transformations)):
            I_b += transformations[I]
            if verbose:
                print("Total intensity of barycenter: ", I_b.sum())

        I_b = 1/len(transformations) * I_b  # Barycenter is average of registered images

        # Resetting the transformations for averaging
        transformations = []

        # Registering a single image onto the current barycenter
        for i in range(len(params)):
            idx = int(i)
            u = params[idx]

            if verbose:
                progress_images_inner = []
                progress_grid_inner = []

            for t2 in range(inner_epochs):
                # forward pass
                optimizer.zero_grad()
                phi_i = xG + u

                # Spacial gradient with boundary conditions
                u_x1 = 1 / dx1 * torch.cat((u[:, 1:, 0] - u[:, :-1, 0], torch.zeros((u.shape[0], 1), device=device, dtype=dtype)),1)
                u_x2 = 1 / dx2 * torch.cat((u[1:, :, 0] - u[:-1, :, 0], torch.zeros((1, u.shape[1]), device=device, dtype=dtype)),0)
                u_y1 = 1 / dx1 * torch.cat((u[:, 1:, 1] - u[:, :-1, 1], torch.zeros((u.shape[0], 1), device=device, dtype=dtype)),1)
                u_y2 = 1 / dx2 * torch.cat((u[1:, :, 1] - u[:-1, :, 1], torch.zeros((1, u.shape[1]), device=device, dtype=dtype)),0)
                u_x = torch.stack((u_x1, u_x2), 2)
                u_y = torch.stack((u_y1, u_y2), 2)

                # Gridsample
                I_i = fixed[idx]
                I_0_phi_i = F.grid_sample(I_i.view(1, 1, N, N), phi_i.unsqueeze(0), align_corners=True).squeeze()

                # Computing the Loss
                jacobian = (1 + u_x1) * (1 + u_y2) - u_x2 * u_y1
                loss_d = ((I_0_phi_i - I_b).pow(2) * dx1 * dx2).sum()
                loss_r = gamma * ((u_x.pow(2).sum(2) + u_y.pow(2).sum(2)) * dx1 * dx2).sum()
                loss = loss_d + loss_r

                # Backward pass
                loss.backward()

                # Recording the loss
                losses[idx].append(loss.item())
                losses_d[idx].append(loss_d.item())
                losses_r[idx].append(loss_r.item())

                # Optimize pass
                optimizer.step()

                # Visualizing the inner loop - for debugging purposes
                if verbose and (t2 in display_its_inner):
                    # I_0_phi_i_ = (I_0_phi_i).detach().cpu().numpy()
                    progress_images_inner.append(I_0_phi_i)
                    # phi_i_ = phi_i.detach().cpu().numpy()
                    progress_grid_inner.append(phi_i)

            # Adding the last registered image onto our transformations to compute the new barycenter
            transformations.append(I_0_phi_i.detach())

            # Visualizing the last registration of the inner loop
            if verbose:
                visualize_registration(I_i, I_b, I_0_phi_i)

            # Quantifying folds in last registered grid
            zero = torch.tensor(0, device=device, dtype=dtype)
            negative_j = torch.where(jacobian < zero, jacobian, zero)
            n_folds = torch.count_nonzero(negative_j).item()
            folds[idx].append(n_folds)

            # Appending progress grids and the progress registration of each image that was averaged
            if t in display_its:
                # phi_i_ = phi_i.detach().cpu().numpy()
                # I_0_phi_i_ = I_0_phi_i.detach().cpu().numpy()
                progress_images_grid[idx].append(phi_i)
                progress_images_reg[idx].append(I_0_phi_i)

            # Visualizing the progress of the inner transformations
            if verbose:
                k = 1
                for j in range(len(display_its_inner)):
                    ax = plt.subplot(2, 3, k)
                    k += 1
                    ax = plot_image_deformation(progress_images_inner[j], progress_grid_inner[j], ax)
                    ax.set_title("t = {}".format(display_its_inner[j]))
                plt.show()

        # Appending the progress images of the barycenter
        if t in display_its:
            # I_b_ = (I_b).detach().cpu().numpy()
            progress_images.append(I_b)

    # Plotting the progress images
    k = 1
    for t in range(len(display_its)):
        ax = plt.subplot(2, 3, k)
        k += 1
        ax.imshow(progress_images[t], vmin=0, cmap="gray")
        ax.set_title("t = {}".format(display_its[t]))
    plt.show()

    # Plotting the progress grids
    for i in range(len(progress_images_grid)):
        k = 1
        for t in range(len(display_its)):
            ax = plt.subplot(2, 3, k)
            k += 1
            phi_i_idx = progress_images_grid[i][t]
            ax = plot_image_deformation(progress_images_reg[i][t], phi_i_idx, ax)
            ax.set_title("t = {}".format(display_its[t]))
        plt.show()

    # Plotting the images' individual losses
    for i in range(len(losses)):
        plt.plot(range(epochs * inner_epochs), losses[i], label="Total Loss")
        plt.plot(range(epochs * inner_epochs), losses_d[i], label="Dissimilarity")
        plt.plot(range(epochs * inner_epochs), losses_r[i], label="Regularizer")
        plt.title("Loss")
        plt.ylabel("SSD")
        plt.xlabel("epochs")
        plt.legend()
        plt.show()

    # Plotting the total loss (sum of the images' losses)
    total_loss = np.array(losses)
    total_loss_d = np.array(losses_d)
    total_loss_r = np.array(losses_r)
    plt.plot(range(epochs * inner_epochs), total_loss.sum(0), label="Sum of losses")
    plt.plot(range(epochs * inner_epochs), total_loss_d.sum(0), label="Dissimilarity")
    plt.plot(range(epochs * inner_epochs), total_loss_r.sum(0), label="Regularizer")
    plt.title("Loss")
    plt.ylabel("SSD")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

    # Plotting the number of folds
    plt.plot(range(epochs), folds[0], label="Folds Image 1")
    plt.plot(range(epochs), folds[1], label="Folds Image 2")
    plt.legend()
    plt.title("Folds in the registration")
    plt.xlabel("Epochs (Outer)")
    plt.ylabel("Number of folds")
    plt.show()

    # Saving the barycenter
    np.save(outpath, progress_images[-1].detach().cpu().numpy())


if __name__ == '__main__':
    main(verbose=True)
