import math
import numpy as np

class Polynomial():
    def __init__(self, coefficients=None):
        if coefficients is not None and type(coefficients) != list:
            raise ValueError(f"Polynomial.__init__(): The type of coefficients ({type(coefficients)}) is not 'list'")
        if coefficients is not None:
            self._coefficients = coefficients
            self._degree = len(self._coefficients) - 1
        else:
            self._coefficients = []
            self._degree = None

    def evaluate(self, x):
        if self._coefficients is None:
            raise ValueError("Polynomial.evaluate(): self._coefficients is None")
        elif len(self._coefficients) == 0:
            raise ValueError("Polynomial.evaluate(): self._coefficients is empty")
        sum = self._coefficients[0]
        for exponent in range(1, self._degree + 1):
            coef = self._coefficients[exponent]
            sum += coef * x ** exponent
        return sum

    def create(self, xy_tuples, degree):  # Create a polynomial with the given (x, y) observations
        self._degree = degree
        if len(xy_tuples) < self._degree + 1:
            raise ValueError(f"Polynomial.create(): len(xy_tuples) ({len(xy_tuples)}) < self.degree + 1 ({self.degree + 1})")
        number_of_observations = len(xy_tuples)
        A = np.zeros((number_of_observations, self._degree + 1), dtype=float)
        b = np.zeros((number_of_observations,), dtype=float)
        for observationNdx in range(number_of_observations):
            x = xy_tuples[observationNdx][0]
            y = xy_tuples[observationNdx][1]
            b[observationNdx] = y
            A[observationNdx, 0] = 1
            for col in range(1, self._degree + 1):
                A[observationNdx, col] = x ** col

        # Least-square solve
        z, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        self._coefficients = z.tolist()

    def evaluate_list(self, xs):
        if not type(xs) == list:
            raise ValueError(f"Polynomial.evaluate_list(): type(xs) ({type(xs)}) is not 'list")
        ys = []
        for x in xs:
            ys.append(self.evaluate(x))
        return ys