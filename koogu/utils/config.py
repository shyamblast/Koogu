
import configparser
from enum import Enum


class Config:

    def __init__(self, cfg_file, sections=None, template=None):
        """
        Provides an object with which one can access settings like cfg_object.SECTION.field

        :param cfg_file: Path to config file.
        :param sections: Name(s) of section(s) that must be extracted. If None, all sections in template will be
            retrieved.
        :param template: The template to process the config file with. In None, default BirdNET-like one will be used.
        """

        if template is None:
            template = _get_default_template()      # Use the default

        if sections is None:
            sections = template.keys()              # Read and return all sections
        elif not isinstance(sections, (tuple, list)):
            sections = [sections]                   # A single section requested, make it a list of one item

        unknown_sections = [s for s in sections if s not in template.keys()]
        if len(unknown_sections) > 0:
            raise ConfigError(cfg_file, None, None, 'Unknown section(s) requested {}'.format(unknown_sections))

        # Read the file contents
        cfg_parser = configparser.ConfigParser(inline_comment_prefixes='#', delimiters='=')
        with open(cfg_file, 'r') as f:
            cfg_parser.read_file(f)

        for section in sections:
            try:
                sec = Config._Section(template[section], cfg_parser[section])
            except ConfigError as exc:
                raise ConfigError(cfg_file, section, exc.field, str(exc))
            except Exception as exc:
                raise ConfigError(cfg_file, section, None, str(exc))
            else:
                setattr(self, section, sec)

    def __repr__(self):
        return '(' + '; '.join(['{:s} {}'.format(k, v) for k, v in vars(self).items()]) + ')'

    class _Section:
        """Internal-use class for handling a single section in a config file"""
        def __init__(self, section_template, section_contents):

            for field_name, field_spec in section_template.items():

                field_val = section_contents.get(field_name, fallback=None)

                if isinstance(field_spec, list) and None in field_spec:  # Field is optional
                    if field_val is None or field_val == '' or field_val.lower() == 'none':
                                                                         # ... and there's no value? that's fine
                        setattr(self, field_name, None)
                        continue
                    else:                                       # ... there was some value?
                        del field_spec[field_spec.index(None)]  # ... remove optional flag for further processing
                elif field_val is None or field_val == '':  # Field isn't optional but there was no value
                    raise ConfigError(None, None, field_name, 'Value unavailable for non-optional field.')

                try:
                    field_val = Config._Section._process_value(field_spec, field_val)
                except ValueError as exc:
                    raise ConfigError(None, None, field_name, str(exc))
                except TypeError as exc:
                    raise ConfigError(None, None, field_name, str(exc))
                except NameError as _:
                    raise ConfigError(None, None, field_name,
                                      'Error evaluating field value as Python code - {}'.format(repr(field_val)))
                except SyntaxError as _:
                    raise ConfigError(None, None, field_name,
                                      'Error evaluating field value as Python code - {}'.format(repr(field_val)))
                else:
                    setattr(self, field_name, field_val)

        def __repr__(self):
            return '[[ ' + ', '.join(['{:s}={}'.format(k, repr(v)) for k, v in vars(self).items()]) + ' ]]'

        @staticmethod
        def _process_value(formats, value):
            """Process RHS values using the provided format specification(s)"""

            if not isinstance(formats, list):
                formats = [formats]     # Treat as a list

            exceptions = []
            for fmt in formats:
                try:
                    if fmt in (int, float, complex, str):
                        retval = fmt(value)

                    elif fmt is bool:
                        retval = bool(eval(value.title()))

                    elif fmt is dict:
                        retval = eval(value)
                        if not isinstance(retval, dict):    # Make sure we're getting a dict
                            raise ValueError('Expected Python dict type; couldn\'t convert value to a dict.')

                    elif isinstance(fmt, tuple):
                        list_content_fmt = fmt[0]
                        retval = eval(value)

                        # Verify and validate contents
                        if not isinstance(retval, (list, tuple)):
                            raise ValueError('"Python list-like" value expected.')

                        if fmt[1] is not None and len(retval) < fmt[1]:
                            raise ValueError('Too few values available. {:s}{:d} expected.'.format(
                                'Minimum ' if fmt[2] is not None and fmt[1] != fmt[2] else '', fmt[1]))
                        if fmt[2] is not None and len(retval) > fmt[2]:
                            raise ValueError('Too many values available. {:s}{:d} expected.'.format(
                                'Maximum ' if fmt[1] is not None and fmt[1] != fmt[2] else '', fmt[2]))

                        if list_content_fmt in (int, float, complex, str):
                            if not all([isinstance(rv, list_content_fmt) for rv in retval]):
                                raise ValueError('List contents expected to be of {} type'.format(list_content_fmt))
                        elif list_content_fmt is not tuple:
                            raise ValueError('The template has an unsupported type: {}.'.format(repr(fmt)))

                    elif issubclass(fmt, Enum):
                        supported_vals = [fm.name for fm in fmt]
                        if value not in supported_vals:
                            raise ValueError('Unsupported value. Supported values are {}.'.format(supported_vals))
                        retval = value

                    else:
                        raise ValueError('The template has an unsupported type: {}.'.format(repr(fmt)))

                except Exception as exc:
                    exceptions.append(exc)
                else:
                    return retval

            if len(exceptions) > 1:
                raise ValueError('Failed to convert {} to any of the types specified in template.'.format(repr(value)))
            elif len(exceptions) == 1:  # There was only one exception, ...
                raise exceptions[0]     # re-raise it for the caller to handle.


class ConfigError(Exception):
    def __init__(self, file_name=None, section_name=None, field_name=None, *args, **kwargs):
        self.file = file_name
        self.section = section_name
        self.field = field_name

        super(ConfigError, self).__init__(*args, **kwargs)

    def __str__(self):
        retstr = ''
        if self.file is not None:
            retstr = '(' + self.file
            if self.section is not None:
                if self.field is not None:
                    retstr += ' [{:s}:{:s}]'.format(self.section, self.field)
                else:
                    retstr += ' [{:s}]'.format(self.section)
            retstr += '): '
        return retstr + str(self.args[0])


def _get_default_template():
    """Default config file template for Koogu-like projects"""
    return {
        'DATA': {
            'audio_fs': [int, float],
            'audio_filterspec': [(tuple, 3, 3), None],
            'audio_clip_length': float,
            'audio_clip_advance': float,
            'spec_win_len': float,
            'spec_win_overlap_prc': float,
            'spec_nfft_equals_win_len': bool,
            'spec_eps': float,
            'spec_bandwidth_clip': [(float, 2, 2), None],
            'spec_type': Enum('spec_type', ['spec', 'spec_db', 'spec_dbfs', 'spec_pcen',
                                            'melspec', 'melspec_db', 'melspec_dbfs', 'melspec_pcen']),
            'spec_num_mels': [int, None]
        },
        'MODEL': {
            'arch': str,
            'arch_params': [dict, None],
#            'fcn_patch_size': [(int, 2, 2), None],
#            'fcn_patch_overlap': [(int, 2, 2), None],
            'dense_layers': [(int, 1, None), None],
            'preproc': [(tuple, 1, None), None]
        },
        'TRAINING': {
            'epochs': int,
            'epochs_between_evals': int,
            'batch_size': int,
            'learning_rate': float,
            'lr_change_at_epochs': [(int, 1, None), None],
            'lr_update_factors': [(float, 2, None), None],
            'optimizer': str,
            'weighted_loss': bool,
            'l2_weight_decay': float,
            'dropout_rate': float,
#            'augmentations_time_domain': [str, None],
#            'augmentations_timefreq_domain': [str, None],
#            'background_infusion_params': [str, None]
        }
    }


def datasection2dict(data_section):

    return {
        'audio_settings': {
            'desired_fs': data_section.audio_fs,
            'clip_length': data_section.audio_clip_length,
            'clip_advance': data_section.audio_clip_advance,
            'filterspec': data_section.audio_filterspec
        },

        'spec_settings': {
            'win_len': data_section.spec_win_len,
            'win_overlap_prc': data_section.spec_win_overlap_prc,
            'nfft_equals_win_len': data_section.spec_nfft_equals_win_len,
            'eps': data_section.spec_eps,
            'bandwidth_clip': data_section.spec_bandwidth_clip,
            'tf_rep_type': data_section.spec_type,
            'num_mels': data_section.spec_num_mels
        }
    }


def log_config(logger, data_cfg=None, model_cfg=None, training_cfg=None,
               **kwargs):

    logger.info('{:=^40s}'.format(' Settings '))

    for section, cfg in zip(['DATA', 'MODEL', 'TRAINING'],
                            [data_cfg, model_cfg, training_cfg]):

        if cfg is None:
            continue

        logger.info('{:-^40s}'.format(' ' + section + ' '))

        for key in sorted([key for key in dir(cfg)
                           if not key.startswith("_")]):
            if isinstance(cfg.__dict__[key], dict):
                logger.info('{0: >25}: {{'.format(key))
                subdict = cfg.__dict__[key]
                for subkey in sorted([subkey for subkey in subdict.keys()]):
                    logger.info('{0: >27} {1: >18}: {2:}'.format(
                        ' ', subkey, subdict[subkey]))
                logger.info('{0:>28}'.format('}'))

            elif isinstance(cfg.__dict__[key], type):
                logger.info('{0: >25}: {1:}'.format(
                    key, cfg.__dict__[key].__name__))

            else:
                logger.info('{0: >25}: {1:}'.format(key, cfg.__dict__[key]))

    # Other additional settings, if any
    # Only log non-None values in kwargs
    logger.info('{:-^40s}'.format(' MISC '))
    for key in sorted([key for key, val in kwargs.items()
                       if val is not None]):

        if isinstance(kwargs[key], dict):
            logger.info('{0: >25}: {{'.format(key))
            subdict = kwargs[key]
            for subkey in sorted([subkey for subkey in subdict.keys()]):
                logger.info('{0: >27} {1: >18}: {2:}'.format(
                    ' ', subkey, subdict[subkey]))
            logger.info('{0:>28}'.format('}'))

        elif isinstance(kwargs[key], type):
            logger.info('{0: >25}: {1:}'.format(key, kwargs[key].__name__))

        else:
            logger.info('{0: >25}: {1:}'.format(key, kwargs[key]))

    logger.info('=====================================')


# if __name__ == '__main__':
#
#     try:
#         cfg = Config('/mnt/datadrive2tb/katydid_stuff/katydid.conf', ['DATA', 'TRAINING'])
#     except ConfigError as exc:
#         print(exc)
#     else:
#         print(cfg)
#         print(cfg.TRAINING.dim_ordering)

