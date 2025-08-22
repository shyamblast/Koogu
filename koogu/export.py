import os
import sys
import shutil
import json
from .model.trained_model import TrainedModel
from .data.raw import Settings
from .utils.config import Config, ConfigError
from zipfile import ZipFile
import argparse


def raven(cfg_file,
          model_name=None, output_prefix=None, model_desc=None,
          threshold=0.9, suppress_classes=None):

    # Defaults
    if model_desc is None:
        model_desc = f'Koogu model packaged using koogu.export.raven v1.0.0'

    models_root, export_root = _get_dirs_from_config(cfg_file)
    trained_model, model_dir = _get_model_and_path(models_root, model_name)

    out_dir = os.path.join(
        export_root, 'Raven',
        ('' if output_prefix is None else output_prefix) +
        os.path.split(model_dir)[-1])

    bandwidth = [0, trained_model.audio_settings['desired_fs'] // 2]
    if trained_model.spec_settings:
        if 'bandwidth_clip' in trained_model.spec_settings:
            bandwidth = trained_model.spec_settings['bandwidth_clip']

    # Create directories
    try:
        os.makedirs(os.path.join(out_dir, 'classes'),
                    exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels'),
                    exist_ok=True)
    except Exception as exc:
        print(f'Unable to create output directories in "{out_dir}": {exc}',
              file=sys.stderr)
        return

    model_config = {
        'specVersion': 1,
        'modelDescription': model_desc,
        'signatures': [
            {
                'signatureName': "with_transformation",
                'modelInputs': [
                    {
                        'inputName': 'inputs',
                        'sampleRate':
                            trained_model.audio_settings['desired_fs'],
                        'inputConfig': ['batch', 'samples']
                    }
                ],
                'modelOutputs': [
                    {
                        'outputName': 'scores',
                        'outputType': 'SCORES'
                    }
                ],
                'semanticKeys': []
            }
        ],
        'modelTypeConfig': {'modelType': 'RECOGNITION'},
        'globalSemanticKeys': trained_model.class_names
    }

    # Copy model files
    shutil.copy(
        os.path.join(model_dir, TrainedModel.saved_model_dirname, 'saved_model.pb'),
        os.path.join(out_dir, 'saved_model.pb')
    )
    shutil.copytree(
        os.path.join(model_dir, TrainedModel.saved_model_dirname, 'variables'),
        os.path.join(out_dir, 'variables'),
        dirs_exist_ok=True
    )

    # Write model config file
    try:
        with open(os.path.join(out_dir, 'model_config.json'), 'w') as fd:
            json.dump(model_config, fd)
    except Exception as exc:
        print(f'Unable to write model config in "{out_dir}": {exc}')
        return

    # Write class info
    if suppress_classes is None or len(suppress_classes) == 0:
        def suppress(_): return 'FALSE'
    else:
        def suppress(class_name):
            return str(
                any([class_name == scn for scn in suppress_classes])).upper()
    try:
        with open(os.path.join(out_dir, 'classes', 'output_classes.csv'),
                  'w') as fd:
            for cn in trained_model.class_names:
                fd.write(f'{cn},{threshold},{bandwidth[0]},{bandwidth[1]},' +
                         f'{suppress(cn)}\n')
    except Exception as exc:
        print('Unable to write output_classes.csv in ' +
              f'"{out_dir}/classes": {exc}'
              )
        return

    # Write label info
    try:
        with open(os.path.join(out_dir, 'labels', 'output_labels.csv'),
                  'w') as fd:
            for cn in trained_model.class_names:
                fd.write(f'{cn},{cn}\n')
    except Exception as exc:
        print(f'Unable to write output_labels.csv in "{out_dir}/labels": {exc}')
        return


def pamguard(cfg_file,
             model_name=None, output_prefix=None, model_desc=None):

    # Defaults
    if model_desc is None:
        model_desc = f'Koogu model packaged using koogu.export.pamguard v1.0.0'

    models_root, export_root = _get_dirs_from_config(cfg_file)
    trained_model, model_dir = _get_model_and_path(models_root, model_name)

    output_path = os.path.join(
        export_root, 'PAMGuard',
        ('' if output_prefix is None else output_prefix) +
        os.path.split(model_dir)[-1] + '.kgu')

    audio_settings = trained_model.audio_settings
    audio_settings_c = Settings.Audio(**audio_settings)

    # Build the list of transforms
    transforms = [
        dict(name='load_audio', params=dict(sr=audio_settings_c.fs))
    ]

    # Add "filtering info" if available
    filter_params = dict()
    if audio_settings['filterspec'] is not None:
        filter_order, filter_critical_freq, filter_type_str = \
            audio_settings['filterspec']

        filter_params['filtermethod'] = 0   # for Butterworth (1=Chebychev)
        filter_params['order'] = filter_order
        if filter_type_str == 'bandpass':
            filter_params['filtertype'] = 2
            filter_params['lowcut'] = filter_critical_freq[0]
            filter_params['highcut'] = filter_critical_freq[1]
        elif filter_type_str == 'highpass':
            filter_params['filtertype'] = 1
            filter_params['lowcut'] = filter_params['highcut'] = \
                filter_critical_freq[0]
        elif filter_type_str == 'lowpass':
            filter_params['filtertype'] = 0
            filter_params['lowcut'] = filter_params['highcut'] = \
                filter_critical_freq[0]
    transforms.append(dict(name='filter_wav', params=filter_params))

    # Add (empty) normalization since the model already handles that step
    transforms.append(dict(name='normalize_wav', params=dict()))

    metadata = dict(
        framework_info=dict(framework='Bespoke'),
        model_info=dict(
            output_shape=[-1, len(trained_model.class_names)],
            input_shape=[-1, audio_settings_c.clip_length, 1, 1]
        ),
        class_info=dict(
            name_class=trained_model.class_names,
            num_class=len(trained_model.class_names)
        ),
        transforms=transforms,
        description=model_desc,
        version_info=dict(version=1),
        seg_size=dict(size_ms=audio_settings['clip_length'] * 1000)
    )

    # Create directory
    try:
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    except Exception as exc:
        print('Unable to create output directory ' +
              f'"{os.path.split(output_path)[0]}": {exc}',
              file=sys.stderr)
        return

    with ZipFile(output_path, 'w') as zf:

        # Copy model files
        zf.write(os.path.join(model_dir, 'koogu/saved_model.pb'),
                 'saved_model.pb')
        vardir = os.path.join(model_dir, 'koogu/variables')
        for f in os.listdir(vardir):
            zf.write(os.path.join(vardir, f), f'variables/{f}')

        # Add metadata
        with zf.open("audio_repr_pg.json", "w") as entry:
            entry.write(json.dumps(metadata, indent=2).encode())


def _get_dirs_from_config(cfg_file):

    # Load config
    try:
        cfg = Config(cfg_file)
    except FileNotFoundError as exc:
        print(f'Error loading config file: {exc.strerror}', file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print(f'Error processing config file: {str(exc)}', file=sys.stderr)
        exit(1)
    except Exception as exc:
        print(f'Error processing config file: {repr(exc)}', file=sys.stderr)
        exit(1)

    return cfg.paths.model, cfg.paths.export


def _get_model_and_path(models_root, model_name=None):

    if model_name is not None:
        model_dir = os.path.join(models_root, model_name)
    else:
        # Find the latest dir (based on name). Assumes all directories
        # correspond to koogu-trained models.
        all_subdirs = []
        for subdir in os.listdir(models_root):
            temp = os.path.join(models_root, subdir)
            if os.path.isdir(temp) and \
                    os.path.exists(
                        os.path.join(temp, TrainedModel.saved_model_dirname)):
                all_subdirs.append(subdir)

        if len(all_subdirs) == 0:
            print('No trained models available in the project.',
                  file=sys.stderr)
            exit(1)

        latest = sorted(all_subdirs)[-1]
        print(f'Latest model "{latest}" will be exported.')

        model_dir = os.path.join(models_root, latest)

    # Load model
    trained_model = TrainedModel(model_dir)

    return trained_model, model_dir


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu.export',
            description='Export trained model for use with other software.')

    parser.add_argument(
        'cfg_file', metavar='<CONFIG FILE>',
        help='Path to config file.')
    parser.add_argument(
        '--model', dest='model_name', metavar='<MODEL>',
        help='Directory name of a trained model instance. If unspecified, '
             'the latest trained model will be exported.')
    parser.add_argument(
        '--prefix', dest='output_prefix', metavar='<PREFIX>',
        help='String prefix to be added to the output directory/file name.')
    parser.add_argument(
        '--desc', dest='model_desc', metavar='<DESCRIPTION>',
        help='Textual description for the model.')

    subparsers = parser.add_subparsers(
        title='Target type', required=True,
        description='The following targets types are supported.')

    p_raven = subparsers.add_parser('raven', help='Raven Pro')
    p_raven.add_argument(
        '--threshold', dest='threshold', metavar='<THRESHOLD>',
        default=0.90,
        help='Set the default detection threshold within Raven Pro.')
    p_raven.add_argument(
        '--suppress', dest='suppress_classes', metavar='<CLASS-NAME>',
        nargs='*',
        help='Instruct Raven Pro to suppress outputs for specified class(es).')
    p_raven.set_defaults(exec_fn=raven)

    p_pmgrd = subparsers.add_parser('pamguard', help='PAMGuard')
    p_pmgrd.set_defaults(exec_fn=pamguard)


if __name__ == '__main__':

    args = cmdline_parser().parse_args()
    args.exec_fn(**{k: getattr(args, k)
                    for k in vars(args) if k != 'exec_fn'})
