from typing import (
    Optional
)

from pathlib import Path
import os

import pandas as pd

from matplotlib import (
    figure
)
"""Pandas I/O

All functions related to I/O operations using Pandas library
"""

def read_csv(filename : str, directory : Optional[str] = './', **kwargs) -> pd.DataFrame:
    """Reads a CSV into a Pandas DataFrame.

    Args:
        filename (str): Name of the file containing the data to be read.
        directory (str, optional): Allows to specify the directory path to use. Defaults to './'.
        kwargs: Other keyword arguments are passed to [`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

    Returns:
        pd.DataFrame: Resulting Pandas DataFrame from the file. 
    """
    return pd.read_csv(Path(directory, filename), **kwargs)

def write_csv(df : pd.DataFrame, filename : str, directory : Optional[str] = './', **kwargs) -> None:
    """Writes a Pandas DataFrame into a CSV file. Directory is created automatically if it doesn't exist.

    Args:
        df (pd.DataFrame): Pandas DataFrame to be written to the file.
        filename (str): Name of the file containing the data to be written to.
        directory (Optional[str], optional): Name of the directory where file is going to be stored. Defaults to './'.
        kwargs: Other keyword arguments are passed to [`pandas.DataFrame.to_csv`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).
    """
    os.makedirs(directory, exist_ok=True)
    df.to_csv(Path(directory, filename), **kwargs)

def read_excel(filename : str, directory : Optional[str] = './', **kwargs) -> pd.DataFrame:
    """Reads an Excel into a Pandas DataFrame.

    Args:
        filename (str): Name of the file containing the data to be read.
        directory (str, optional): Allows to specify the directory path to use. Defaults to './'.
        kwargs: Other keyword arguments are passed to [`pandas.read_excel`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html).

    Returns:
        pd.DataFrame: Resulting Pandas DataFrame from the file. 
    """
    return pd.read_excel(Path(directory, filename), **kwargs)

"""Plot I/O

All functions related to I/O operations for plots used in Matplotlib, Seaborn, Worldcloud and other plotting libraries. 
"""

def save_figure(figure : figure.Figure, filename : str, directory : Optional[str] = './', **kwargs) -> None:
    """Saves the figure into the file specified. Directory is created automatically if it doesn't exist.

    Args:
        figure (figure.Figure): Matplotlib Figure to be saved.
        filename (str): Name of the file where figure is to wrote to.
        directory (Optional[str], optional): Name of directory where file is going to be stored. Defaults to './'.
        kwargs: Other keyword arguments are passed to [`matplotlib.figure.Figure.savefig`](https://matplotlib.org/stable/api/figure_api.html?highlight=savefig#matplotlib.figure.Figure.savefig).
    """
    os.makedirs(directory, exist_ok=True)
    figure.savefig(Path(directory, filename), **kwargs)