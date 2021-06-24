"""
List of tasks to be used by the Invoke library

e.g.

    nohup inv splice-test &
    nohup inv rw-test &
    inv mount bbfs,sshfs 256 256
    inv unmount --prompt-for-sudo-password bbfs,sshfs

"""

from itertools import product
from subprocess import run


def flatten(nested: dict) -> list:
    """
    Convert dictionary with a list of values for each key to a list of
        dictionaries with single value for each key

    e.g.
        {
            'block_size': [4, 16, 256, 1024],
            'file_size': 128,
        }
        to
        [
            {'block_size': 4,
             'file_size': 128},
            {'block_size': 16,
             'file_size': 128},
            {'block_size': 256,
             'file_size': 128},
            {'block_size': 1024,
             'file_size': 128}
        ]
    """
    for key in nested.keys():
        value = nested[key]
        if not isinstance(value, list):
            nested[key] = [value]
        elif not value:
            continue

    return [dict(zip(nested.keys(), vals)) for vals
            in product(*nested.values())]


def synthesize_data():
    configurations = [
        {
            'N': [128, 256, 512, 1024, 2048],
            'k': [20, 30, 40],
            'maxk': [100, 200],
            'mu': [0.1, 0.2, 0.5]
        }
    ]
    prefix_templ = '-prefix {N}_{k}_{maxk}_{mu}'
    run_params = list()
    [run_params.extend(flatten(conf)) for conf in configurations]
    cli_template = 'bin/benchmark {option_str} {prefix_option}'
    for param in run_params:
        prefix_option = prefix_templ.format(
            **param
        )
        prefix_option = prefix_option.replace('0.', '')
        option_str = ' '.join([
            f'-{k} {v}' for k, v in param.items()
        ])
        cli = cli_template.format(
            option_str=option_str,
            prefix_option=prefix_option
        )
        print('Running {:s}'.format(cli))
        complete = run(
            cli.split()
        )


if __name__ == '__main__':
    synthesize_data()
