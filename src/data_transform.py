import numpy as np


def custom_transform(data):
    """
    Transform the `spiral.csv` data such that it can be more easily classified.

    To pass test_custom_transform_hard, your transformation should create at
    most three features and should allow a LogisticRegression model to achieve
    at least 90% accuracy.

    You can use free_response.q2.visualize_spiral() to visualize the spiral
    as we give it to you, and free_response.q2.visualize_transform() to
    visualize the 3D data transformation you implement here.

    Args:
        data: a Nx2 matrix from the `spiral.csv` dataset.

    Returns:
        A transformed data matrix that is (more) easily classified.
    """
    theta = np.apply_along_axis(lambda x: np.arctan(x[1]/x[0]), arr=data, axis=1)
    r = np.apply_along_axis(lambda x: np.sqrt(x[0]**2 + x[1]**2), arr=data, axis=1)
    x = data[:,0] / np.cos(theta)
    x /= np.sign(x)
    x = x[:, np.newaxis]
    y = data[:,1] / np.sign(theta)
    y = y[:, np.newaxis]
    r = r[:, np.newaxis]
    remain = (x % np.pi) < np.pi/2
    y[remain.flatten()] = y[remain.flatten()] * -1
    remain = (x % (np.pi*2)) < np.pi
    y[remain.flatten()] = y[remain.flatten()] * -1
    return np.append(x, y, axis=1)

