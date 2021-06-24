"""
Data preparation script
"""

import click
from pathlib import Path

from data import load_data, supported_exts


def scan_processed(root: Path):
    processed = list()
    for f in root.iterdir():
        if f.is_file() and f.suffix == '.pkl':
            processed.append(f.stem)
    return processed


@click.command()
@click.option('--overwrite/--skip-overwrite',
              default=False, help='Overwrite or skip existing datasets')
def data_prep(overwrite: bool):
    data_root = Path(__file__).parent.parent.parent / 'data'
    processed_root = data_root / 'datasets'
    existing_datasets = scan_processed(processed_root) if not overwrite else list()
    for f in (data_root / 'raw').iterdir():
        if f.suffix not in supported_exts.keys():
            print(f'File {str(f)} not supported')
            continue
        if not overwrite and f.stem in existing_datasets:
            print(f'Dataset {f.stem} already exist')
            continue
        try:
            df = load_data(
                f
            )
        except Exception:
            print(f'Fail to parse {str(f)}')
            continue
        df.to_pickle(
            str(processed_root / f'{f.stem}.pkl')
        )


if __name__ == '__main__':
    data_prep()
