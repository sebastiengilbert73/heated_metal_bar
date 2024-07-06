import os
import logging
import pandas as pd
import argparse
import numpy as np
import math
import fourier.series
import matplotlib.pyplot as plt
import cv2
import imageio

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    alpha,
    duration,
    numberOfTimesteps,
    initialTemperatureProfile
):
    logging.info("both_ends_fixed.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the initial temperature profile
    x_u_df = pd.read_csv(initialTemperatureProfile)
    x_u = x_u_df.values
    length = x_u[-1, 0]
    number_of_points = x_u.shape[0]
    logging.info(f"length = {length}; number_of_points = {number_of_points}")
    xs = x_u[:, 0]
    u0 = x_u[:, 1]

    # Compute the Fourier series of u0.
    # Since both ends are fixed, we'll use an odd half-range expansion
    # u0_hat(x) = u0(x) - Cx - D
    C = (u0[-1] - u0[0])/length
    D = u0[0]
    u0_hat = u0 - C * xs - D

    # Plot u0, u0_hat, and Cx + D
    fig, ax = plt.subplots()
    ax.plot(xs, u0, label='u0', linewidth=3)
    ax.plot(xs, u0_hat, label='û0', linewidth=3)
    ax.plot(xs, C * xs + D, label='Cx + D', linewidth=3)

    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('u (arbitrary units)')
    ax.set_xlim((0, length))
    ax.set_ylim((-40, 80))
    ax.legend()
    plt.show()

    expander = fourier.series.Expander(length, 'odd')
    a, b = expander.coefficients(u0_hat)
    reconstructed_u0_hat = expander.reconstruct(a, b, len(xs))

    # Plot u0_hat and its reconstruction
    fig, ax = plt.subplots()
    ax.plot(xs, u0_hat, label='û0', linewidth=3)
    ax.plot(xs, reconstructed_u0_hat, label='reconstructed û0', linewidth=1)

    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('u (arbitrary units)')
    ax.set_xlim((0, length))
    ax.set_ylim((-40, 10))
    ax.legend()
    plt.show()

    # Compute the lambda_n
    lambda_n = []
    for n in range(len(a)):
        lambda_n.append(-alpha * n ** 2 * math.pi ** 2 / length ** 2)

    u_x_t = np.zeros((numberOfTimesteps, number_of_points))
    delta_t = duration / (numberOfTimesteps - 1)
    ts = np.arange(0, duration + delta_t / 2, delta_t)
    images = []
    for t_ndx in range(len(ts)):
        t = ts[t_ndx]
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            sum = C * x + D
            for n in range(1, len(lambda_n)):
                sum += np.exp(lambda_n[n] * t) * b[n] * math.sin(n * math.pi * x / length)
            u_x_t[t_ndx, x_ndx] = sum
        fig, ax = plt.subplots()
        ax.plot(xs, u_x_t[t_ndx, :], label=f"t = {t:.3} s")
        ax.legend(loc='lower right')
        ax.grid(True)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('u (arbitrary units)')
        ax.set_xlim((0, length))
        ax.set_ylim((-40, 80))
        img_filepath = os.path.join(outputDirectory, "fig.png")
        plt.savefig(img_filepath)
        plt.close(fig)
        image = cv2.imread(img_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    # Plot the solution
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xs, ts, u_x_t, cmap='viridis')
    ax.axis([0, length, 0, duration])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('t (s)')
    fig.colorbar(c, ax=ax)
    plt.show()

    gif_filepath = os.path.join(outputDirectory, "animation.gif")
    imageio.mimsave(gif_filepath, ims=images, loop=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_both_ends_fixed'",
                        default="./output_both_ends_fixed")
    parser.add_argument('--alpha', help="The thermal diffusivity, in m^2/s. Default: 0.0001", type=float,
                        default=0.0001)
    parser.add_argument('--duration', help="The duration of the simulation, in seconds. Default: 100.0", type=float,
                        default=100.0)
    parser.add_argument('--numberOfTimesteps', help="The number of timesteps. Default: 101", type=int, default=101)
    parser.add_argument('--initialTemperatureProfile',
                        help="The csv file giving the initial temperature profile. Default: './initial_both_ends_fixed.csv'",
                        default="./initial_both_ends_fixed.csv")
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.alpha,
        args.duration,
        args.numberOfTimesteps,
        args.initialTemperatureProfile
    )