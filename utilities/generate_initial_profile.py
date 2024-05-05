import os
import logging
import pandas as pd
import argparse
import numpy as np
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    length,
    numberOfPoints
):
    logging.info("generate_initial_profile.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    delta_x = length/(numberOfPoints - 1)
    xs = np.arange(0, length + delta_x/2, delta_x)
    x_u = np.zeros((len(xs), 2), float)
    for x_ndx in range(len(xs)):
        x = xs[x_ndx]
        x_u[x_ndx, 0] = x
        x_u[x_ndx, 1] += 0.5 - 0.3 * math.cos(math.pi * x/length)
        if x < 0.03:
            x_u[x_ndx, 1] = 1.0
        elif x >= 0.05 and x < 0.2:
            x_u[x_ndx, 1] += 0.1 * math.sin(2 * math.pi * (x - 0.05)/0.05)
        elif x >= 0.2 and x < 0.28:
            x_u[x_ndx, 1] += 0.1 * math.sin(2 * math.pi * (x - 0.2) / 0.0125)

    x_y_df = pd.DataFrame({'x': x_u[:, 0], 'u': x_u[:, 1]})
    x_y_df.to_csv(os.path.join(outputDirectory, "x_u.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_generate_initial_profile'",
                        default="./output_generate_initial_profile")
    parser.add_argument('--length', help="The bar length, in m. Default: 0.30", type=float, default=0.30)
    parser.add_argument('--numberOfPoints', help="The number of spatial sample points. Default: 201", type=int, default=201)
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.length,
        args.numberOfPoints
    )