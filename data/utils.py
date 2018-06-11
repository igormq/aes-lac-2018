import argparse


def get_argparse(target_dir, fs=16e3, files_to_download=None, min_duration=1, max_duration=15):
    parser = argparse.ArgumentParser('Processes and downloads a dataset.')
    parser.add_argument('--target-dir', default=target_dir, type=str, help='Directory to store the dataset.')
    parser.add_argument('--fs', '--sample-rate', default=fs, type=int, help='Sample rate')
    parser.add_argument(
        '--files-to-download', default=files_to_download, nargs='+', type=str, help='list of file names to download')
    parser.add_argument(
        '--min-duration',
        default=min_duration,
        type=int,
        help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
    parser.add_argument(
        '--max-duration',
        default=max_duration,
        type=int,
        help='Prunes training samples longer than the max duration (given in seconds, default 15)')

    return parser
