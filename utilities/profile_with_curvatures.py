import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("profile_with_curvatures.main()")

    number_of_points = 101
    L = 0.3
    u = np.zeros((number_of_points))
    d2u_dx2 = np.zeros((number_of_points))
    delta_x = L / (number_of_points - 1)
    xs = np.arange(0, L + delta_x/2, delta_x)
    for x_ndx in range(len(xs)):
        x = xs[x_ndx]
        u_local = 0
        d2u_dx2_local = 0
        if x < 0.07:
            u_local = 20 + 50.0 * x
        elif x < 0.15:
            u_local = 24.0 - 1250 * (x - 0.09)**2
            d2u_dx2_local = -2500
        else:
            u_local = 12 + 750 * (x - 0.25)**2
            d2u_dx2_local = 1500
        u[x_ndx] = u_local
        d2u_dx2[x_ndx] = d2u_dx2_local

    # Subsample the arrows
    """subsampled_xs = []
    subsampled_us = []
    subsampled_d2u_dx2 = []
    for x_ndx in range(0, len(xs), 5):
        subsampled_xs.append(xs[x_ndx])
        subsampled_us.append(u[x_ndx])
        subsampled_d2u_dx2.append(d2u_dx2[x_ndx])
    """

    fig, ax = plt.subplots()
    ax.plot(xs, u)
    for x_ndx in range(0, len(xs), 5):
        plt.arrow(xs[x_ndx], u[x_ndx], 0, 0.001 * d2u_dx2[x_ndx], width=0.002, length_includes_head=True, head_width=0.01,
                  head_length=0.5, color='red')
        #ax.annotate("", xy=(0, d2u_dx2[x_ndx]), xytext=(xs[x_ndx], u[x_ndx]), arrowprops=dict(arrowstyle="->"))
    #plt.arrow(np.array(subsampled_xs), np.array(subsampled_us), np.zeros_like(subsampled_xs), np.array(subsampled_d2u_dx2))
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('Temperature')
    plt.show()

if __name__ == '__main__':
    main()