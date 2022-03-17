import matplotlib.pyplot as plt
import numpy as  np
import torch

def plot_latent_space(vae, n=30, figsize=15, device='cuda', save_root="./fig/", save_plot=False, epoch=1):
    # display [n, n] 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size*n, digit_size*n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xj in enumerate(grid_x):
            z_sample = np.array([[xj, yi]])
            x_decoded = vae.decoder(torch.Tensor(z_sample).to(device)).to('cpu').data.numpy() # [z_sample.size(0), 784]
            x_decoded = x_decoded.reshape(digit_size, digit_size)
            figure[i*digit_size: (i+1)*digit_size, j*digit_size:(j+1)*digit_size] = x_decoded
    
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n* digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    if save_plot:
        plt.savefig(save_root+"latent_space_epoch"+str(epoch)+".png")
    plt.show()

def plot_label_clusters(vae, data, labels, device, save_root="./fig/", save_plot=False, epoch=1):
    # display a 2D plot of the digit classes in the latent space
    mu, log_var = vae.encoder(data.to(device))
    z  = vae.sampling(mu, log_var)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0].to('cpu').data.numpy(), z[:, 1].to('cpu').data.numpy(), c=labels.to('cpu').data.numpy())
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if save_plot:
        plt.savefig(save_root+"label_clusters_epoch"+str(epoch)+".png")
    plt.show()
