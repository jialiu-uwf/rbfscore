"""
Data module
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat


def load_mat(path: Path, mat_name: str = 'A'):
    """
    Load mat format data
    mat is saved in Matlab
    convert all non-zero values to 1
    Will need a name of the desired matrix

    Parameters
    ----------
    path
        path to the data file
    mat_name
        name of the matrix

    Returns
    -------
    a DataFrame holding the data
    """
    mat = loadmat(str(path))[mat_name]
    mat[mat != 0] = 1
    return pd.DataFrame(mat, dtype=np.int8)


def load_txt(path: Path):
    """
    Load csv format data
    Assuming no header and tab as delimiter

    Parameters
    ----------
    path
        path to the data file

    Returns
    -------
    a DataFrame holding the data
    """
    return load_csv(path, '\t')


def load_csv(path: Path, delimiter: str = ','):
    """
    Load csv format data
    Assuming no header

    Parameters
    ----------
    path
        path to the data file
    delimiter
        delimiter used in the csv file

    Returns
    -------
    a DataFrame holding the data
    """
    return pd.read_csv(str(path), header=None, delimiter=delimiter, dtype=np.float_).dropna(axis=1).astype('int8')


def load_data(data_path: Path):
    """
    Load data from a file

    Parameters
    ----------
    data_path : pathlib.Path
        path to a file

    Returns
    -------
    a Pandas DataFrame
    """

    if not data_path.is_file():
        raise FileNotFoundError(f'{str(data_path.absolute())} does not exist!')
    ext = data_path.suffix.lower()
    handlers = {
        '.mat': load_mat,
        '.txt': load_txt,
        '.csv': load_csv
    }
    if ext not in handlers:
        raise ValueError(f'The file extension {ext[1:]} is not supported!')
    return handlers[ext](data_path)
