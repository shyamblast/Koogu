import argparse
from . import explore, prepare, train_and_eval, inference, assess, export


def cmdline_parser():

    parser = argparse.ArgumentParser(
        prog='koogu',
        description='Command-line interface for Koogu')  # TODO: add better desc

    subparsers = parser.add_subparsers(
        title='operations', required=True,
        description='The following operations are supported.')

    # cmd_new = subparsers.add_parser(
    #     'new',
    #     help='Create a new project.')

    cmd_explore = subparsers.add_parser(
        'explore',
        help='Explore datasets.')
    explore.cmdline_parser(cmd_explore)

    cmd_prepare = subparsers.add_parser(
        'prepare',
        help='Prepare audio data for training.')
    prepare.cmdline_parser(cmd_prepare)

    cmd_train = subparsers.add_parser(
        'train',
        help='Train a model using prepared inputs.')
    train_and_eval.cmdline_parser(cmd_train)

    cmd_assess = subparsers.add_parser(
        'assess',
        help='Assess recognition performance of trained model.')
    assess.cmdline_parser(cmd_assess)

    cmd_recognize = subparsers.add_parser(
        'recognize',
        help='Perform recognition using trained model.')
    inference.cmdline_parser(cmd_recognize)

    cmd_export = subparsers.add_parser(
        'export',
        help='Export trained model for use with other software.')
    export.cmdline_parser(cmd_export)

    return parser


def create_starter_config_file(outpath):
    with open(outpath, 'w') as f:
        f.write("""
        """)
    # TODO: yet to implement


if __name__ == '__main__':

    args = cmdline_parser().parse_args()

    args.exec_fn(**{k: getattr(args, k)
                    for k in vars(args) if k != 'exec_fn'})
