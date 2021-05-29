from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# functions
def multiquadric(mat, c):
    """
    multiquadric
    """
    return np.sqrt(mat ** 2 + c ** 2)


def inv_multiquadric(mat, c):
    """
    inverse multiquadric
    """
    return 1 / np.sqrt(mat ** 2 + c ** 2)


def gaussian(mat, c):
    """
    Gaussian
    """
    return np.exp(-mat ** 2 / c ** 2)


def distance_matrix(data, centers):
    """
    Distance matrix
    """
    m, _ = data.shape
    n, s = centers.shape
    dm = np.zeros((m, n))
    for i in range(s):
        assert data[:, i].shape == (m,)
        assert centers[:, i].shape == (n,)
        v1 = data[:, i].reshape(m, 1)
        v2 = centers[:, i].reshape(n, 1)
        dm += np.square(
            # matlib.repmat(v1, 1, n) - matlib.repmat(v2.T, m, 1)
            np.tile(v1, (1, n)) - np.tile(v2.T, (m, 1))
        )
    dm = np.sqrt(dm)
    return dm


def rbf(c, data, rbf_fn):
    """
    RBF function on matrix
    """
    n = data.shape[0]
    x = np.linspace(0, 1, n).reshape(n, 1)
    assert len(x.shape) == 2 and x.shape[1] == 1, "x should be a n x 1 vector"
    dm = distance_matrix(x, x)
    return np.multiply(rbf_fn(dm, c), data)


def load_mat(path, mat_name='A'):
    return loadmat(path)[mat_name]


def rbf_param_opt():
    """
    RBF parameters optimization

    to optimize:
        c
        rbf function

    metric:
        condition number

    """
    # configurations
    c_steps = 50
    c_start = 0.01
    c_end = 0.1
    rbf_fns = {
        'MQ': multiquadric,
        'iMQ': inv_multiquadric,
        'Gaussian': gaussian
    }

    proj_root = Path('../..')
    result_root = proj_root / 'data/results'

    # data loading
    data_root = proj_root / 'data/datasets'
    for f in data_root.iterdir():
        if f.is_file() and f.name[-4:] == '.mat':
            ds_name = f.stem
            data = load_mat(f.absolute().__str__())
            assert data.shape[0] == data.shape[1]

            # computation
            c_trials = np.linspace(c_start, c_end, c_steps)
            for rbf_fn_name in rbf_fns:
                rbf_fn = rbf_fns[rbf_fn_name]

                conditions = list()
                for c in c_trials:
                    adj_mat = rbf(c, data, rbf_fn)
                    conditions.append(
                        np.linalg.cond(adj_mat)
                    )
                result = np.vstack(
                    (c_trials, np.asarray(conditions))
                )
                result = pd.DataFrame(
                    result,
                    index=['c', 'cond']
                )
                result.T.plot(x='c', y='cond', logy=True)
                plt.xlabel('c value')
                plt.ylabel('Condition number (log scale)')
                plt.title(
                    f'Condition numbers over c values, {rbf_fn_name} function'
                )
                plot_file = result_root / f'{ds_name}_cond_plot_{rbf_fn_name}.pdf'
                plt.savefig(str(plot_file))
                plt.close()
                print(f'finished with dataset {ds_name} with {rbf_fn_name}')


if __name__ == "__main__":
    rbf_param_opt()
