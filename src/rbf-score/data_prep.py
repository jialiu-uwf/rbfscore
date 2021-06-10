"""
Data preparation script
"""


from pathlib import Path
from data import *
import pandas as pd


def main():
    data_root = Path(__file__).parent.parent.parent / 'data'
    output_root = data_root / 'datasets'
    for f in (data_root / 'raw').iterdir():
        if f.suffix not in ['.mat', '.txt', '.csv']:
            print(f'file {str(f)} not supported')
            continue
        try:
            df = load_data(
                f
            )
        except Exception:
            print(f'Fail to parse {str(f)}')
            continue
        df.to_pickle(
            str(output_root / f'{f.stem}.pkl')
        )


if __name__ == '__main__':
    main()