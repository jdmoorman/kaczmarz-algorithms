import numpy as np
import pytest

import kaczmarz

np.random.seed(0)

quantile_strategies = []
for name in dir(kaczmarz):
    if name.startswith("_"):
        continue
    if "Quantile" not in name:
        continue
    attr = getattr(kaczmarz, name)
    if attr == kaczmarz.Base:
        continue
    if not issubclass(attr, kaczmarz.Base):
        continue
    quantile_strategies.append(attr)


def get_systems():
    examples = []

    A = np.arange(1, 11).reshape(-1, 1)  # [[1], [2], ...]
    x = np.array([1])
    examples.append((A, x))

    A = np.random.normal(size=(10, 2))
    x = np.array([1, 1])
    examples.append((A, x))

    return examples


class Sample100(kaczmarz.SampledQuantile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_samples=10, **kwargs)


class Window100(kaczmarz.WindowedQuantile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, window_size=100, **kwargs)


quantile_strategies.append(Sample100)
quantile_strategies.append(Window100)


@pytest.mark.parametrize("A,x_exact", get_systems())
@pytest.mark.parametrize("QuantileStrategy", quantile_strategies)
def test_corrupted_systems(A, x_exact, QuantileStrategy):
    maxiter = 100  # 100 iterations should be plenty for nice systems.
    # TODO: Add a proper convergence criterion for corrupted systems.
    b = A @ x_exact
    b[0] += 100
    x_approx = QuantileStrategy.solve(A, b, maxiter=maxiter, quantile=0.7)

    residual = b - A @ x_approx

    tol = 1e-5

    assert residual[0] > 99
    assert np.linalg.norm(residual[1:]) < tol
