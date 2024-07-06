import logging
import pandas as pd
import argparse
import ast
import polynomials
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    numberOfPoints,
    points
):
    logging.info("profile_polynomial.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    x0 = points[0][0]
    xL = points[-1][0]
    poly = polynomials.Polynomial()
    poly.create(points, len(points) - 1)
    delta_x = (xL - x0)/(numberOfPoints - 1)
    xs = np.arange(x0, xL + delta_x/2, delta_x)
    ys = poly.evaluate_list(xs.tolist())

    output_filepath = os.path.join(outputDirectory, "profile.csv")
    with open(output_filepath, 'w') as output_file:
        output_file.write("x,u\n")
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            y = ys[x_ndx]
            output_file.write(f"{x},{y}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_profile_polynomial'",
                        default="./output_profile_polynomial")
    parser.add_argument('--numberOfPoints', help="The number of points. Default: 201", type=int, default=201)
    parser.add_argument('--points', help="The list of points where the polynomial is forced to pass. Default: '[(0, 30), (0.08, 10), (0.15, 45), (0.23, 40), (0.3, 70)]'",
                        default='[(0, 30), (0.08, 10), (0.15, 45), (0.23, 40), (0.3, 70)]')
    args = parser.parse_args()

    args.points = ast.literal_eval(args.points)
    main(
        args.outputDirectory,
        args.numberOfPoints,
        args.points
    )