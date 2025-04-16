
import configparser
from enum import Enum
import os
import abc

# Needed for validating some enumerated entries
from ..data import annotations
from ..model import architectures


class Config:
  """
  A configuration handler for managing Koogu projects.
  Provides an object with which one can access project settings in
  dot-notation format (i.e., cfg_object.section.field).

  :param cfg_file: Path to config file.
  :param *sections: Names of sections to load.

  Returned cfg_object always contains contents of 'paths' section.
  """

  def __init__(self, cfg_file, *sections):

    # Read the file contents
    cfg_parser = configparser.ConfigParser(inline_comment_prefixes='#',
                                           delimiters='=')
    with open(cfg_file, 'r') as f:
      cfg_parser.read_file(f, source=cfg_file)

    section = 'paths'
    if section not in cfg_parser:       # This is a mandatory section
      raise ConfigError(cfg_file, None, None,
                        f'Required section \'paths\' is missing.')

    template, default_reldirs = Config._get_template(section)
    try:
      self.paths = _Section(template, cfg_parser[section], cfg_file, section)
    except Exception as exc:
      raise ConfigError(cfg_file, section, None, str(exc))

    # Process overriding settings
    for field, default in default_reldirs.items():
      if getattr(self.paths, field, None) is None:
        setattr(self.paths, field,
                os.path.join(self.paths.project_root, default))

    for section in sections:            # Read and return all requested sections
      template = Config._get_template(section)

      if section not in cfg_parser:     # Section missing
        # Complain if it had any mandatory field(s)
        if not all([(isinstance(rhs, list) and None in rhs)
                    for _, rhs in template.items()]):
          raise ConfigError(cfg_file, None, None,
                            f'Required section [{section}] is missing.')
        else:
          cfg_parser.add_section(section)   # Create dummy empty section

      try:
        sec = _Section(template, cfg_parser[section], cfg_file, section)
      except ConfigError as exc1:
        raise exc1
      except Exception as exc2:
        raise ConfigError(cfg_file, section, None, str(exc2))
      else:
        parent_node, leaf_node_name = self._self_or_subsection(section)
        setattr(parent_node, leaf_node_name, sec)

  def _self_or_subsection(self, section):
    # Successively add subsections (instances of _Section) if and as needed

    parts = section.split('.')

    parent = self
    for part in parts[:-1]:
      if not hasattr(parent, part):
        # Create empty section
        setattr(parent, part, _Section())
      parent = getattr(parent, part)

    return parent, parts[-1]

  @staticmethod
  def _get_template(section):
    """
    Default config template for Koogu projects.
    Specify required section name (str) as parameter.
    """

    # 'paths' section is universal for all commands
    # Returns 2 values:
    #   section template dict
    #   default relative (to project_root) folder/filenames dict
    if section == 'paths':
      return dict(
          project_root=str,
          # Setting all fields below to be optional. It's the calling processes'
          # responsibility to perform any necessary checks.
          train_audio=[str, None],
          train_annotations=[str, None],
          train_audio_annotations_map=[str, None],
          training_samples=[str, None],
          test_audio=[str, None],
          test_annotations=[str, None],
          test_audio_annotations_map=[str, None],
          test_detections=[str, None],
          model=[str, None],
          logs=[str, None]
        ), dict(
          train_audio='train_audio',
          train_annotations='train_annotations',
          train_audio_annotations_map='train_audio_annot_map.csv',
          training_samples='prepared_clips',
          test_audio='test_audio',
          test_annotations='test_annotations',
          test_audio_annotations_map='test_audio_annot_map.csv',
          test_detections='test_detections',
          model='model',
          logs='logs'
        )

    # For all other sections, only one value is returned

    if section == 'data.audio':
      return dict(
        desired_fs=[PosInt, PosFloat],
        filterspec=[(tuple, 3, 3), None],
        clip_length=PosFloat,
        clip_advance=PosFloat
      )

    if section == 'data.spec':
      return dict(
        win_len=PosFloat,
        win_overlap_prc=NonNegFrac,
        nfft_equals_win_len=[bool, None],
        eps=[PosFloat, None],
        bandwidth_clip=[(PosFloat, 2, 2), None],
        type=[
          Enum('spec_type', ['spec', 'spec_db', 'melspec', 'melspec_db']),
          None
        ],
        num_mels=[PosInt, None]
      )

    if section == 'data.annotations':
      return dict(
        annotation_reader=[Enum('reader', annotations.__all__), None],
        desired_labels=[(str, 1, None), None],
        remap_labels_dict=[dict, None],
        raven_label_column_name=[str, None],
        raven_default_label=[str, None]
      )

    if section == 'model':
      return dict(
        architecture=[Enum('arch', architectures.__all__), None],
        architecture_params=[dict, None]
      )

    if section == 'train':
      return dict(
        epochs=PosInt,
        epochs_between_evals=[PosInt, None],
        batch_size=[PosInt, None],
        learning_rate=[PosFloat, None],
        lr_change_at_epochs=[(PosInt, 1, None), None],
        lr_update_factors=[(PosFloat, 2, None), None],
        optimizer=[str, None],
        weighted_loss=[bool, None],
        l2_weight_decay=[PosFloat, None],
        dropout_rate=[PosFrac, None]
      )

    # -- Optional sections --

    if section == 'prepare':
      return dict(
        negative_class=[str, None],
        min_annotation_overlap_fraction=[PosFrac, None],
        max_nonmatch_overlap_fraction=[NonNegFrac, None],
        attempt_salvage=[bool, None]
      )

    if section == 'assess':
      return dict(
        thresholds=[PosFrac, (NonNegFrac, 1, None), None],
        min_gt_coverage=[PosFrac, None],
        min_det_usage=[PosFrac, None],
        squeeze_min_dur=[PosFloat, None]
      )


class _Section:
  """Internal-use class for handling a single section from a config file"""

  def __init__(self, section_template=None, section_contents=None,
               filepath=None, section_name=None):

    if section_template is None:
      return

    for field_name, field_spec in section_template.items():

      field_val = section_contents.get(field_name, fallback=None)

      # Is field optional?
      if isinstance(field_spec, list) and None in field_spec:

        # ... and there's no value? that's fine.
        if field_val is None or field_val == '' or field_val.lower() == 'none':
          setattr(self, field_name, None)
          continue

        else:       # ... there was some value?
          # Remove optional flag for further processing
          del field_spec[field_spec.index(None)]

      elif field_val is None or field_val == '':
        # Field isn't optional but there was no value
        raise ConfigError(
          filepath, section_name, field_name,
          'Value unavailable for non-optional field.')

      try:
        field_val = _Section._process_value(field_spec, field_val)
      except ValueError as exc1:
        raise ConfigError(filepath, section_name, field_name, str(exc1))
      except TypeError as exc2:
        raise ConfigError(filepath, section_name, field_name, str(exc2))
      except NameError as _:
        raise ConfigError(
          filepath, section_name, field_name,
          f'Error evaluating field value as Python code - {repr(field_val)}')
      except SyntaxError as _:
        raise ConfigError(
          filepath, section_name, field_name,
          f'Error evaluating field value as Python code - {repr(field_val)}')
      else:
        setattr(self, field_name, field_val)

  def as_dict(self, skip_invalid=None):
    if skip_invalid:
      return {
        k: v.as_dict() if isinstance(v, _Section) else v
        for k, v in vars(self).items()
        if v is not None
      }
    else:
      return {
        k: v.as_dict() if isinstance(v, _Section) else v
        for k, v in vars(self).items()
      }

  @staticmethod
  def _process_value(formats, value):
    """Process RHS values using the provided format specification(s)"""

    if not isinstance(formats, list):
      formats = [formats]     # Treat as a list

    exceptions = []
    for fmt in formats:
      try:
        if fmt in (int, float, complex, str):   # builtin types
          retval = fmt(value)

        elif fmt is bool:                       # bool
          retval = bool(eval(value.title()))

        elif fmt is dict:                       # dict
          retval = eval(value)
          # Make sure we're getting a dict
          if not isinstance(retval, dict):
            raise ValueError(
              'Expected Python dict type; ' +
              'couldn\'t convert value to a dict.')

        elif isinstance(fmt, tuple):            # tuple
          list_content_fmt = fmt[0]
          retval = eval(value)

          # Verify and validate contents
          if not isinstance(retval, (list, tuple)):
            raise ValueError('"Python list-like" value expected.')

          if fmt[1] is not None and len(retval) < fmt[1]:
            raise ValueError(
              'Too few values available. {:s}{:d} expected.' .format(
                'Minimum ' if (fmt[2] is not None and fmt[1] != fmt[2]) else '',
                fmt[1]))
          if fmt[2] is not None and len(retval) > fmt[2]:
            raise ValueError(
              'Too many values available. {:s}{:d} expected.'.format(
                'Maximum ' if (fmt[1] is not None and fmt[1] != fmt[2]) else '',
                fmt[2]))

          if list_content_fmt in (int, float, complex, str):
            if not all([isinstance(rv, list_content_fmt) for rv in retval]):
              raise ValueError(
                f'List contents expected to be of {list_content_fmt} type.')
          elif issubclass(list_content_fmt, _ConstrainedNumeric):
            # Attempt converting all elements
            for rv in retval: list_content_fmt.check(rv)
          elif list_content_fmt is not tuple:
            raise ValueError(
              f'The template has an unsupported type: {repr(fmt)}.')

        elif issubclass(fmt, _ConstrainedNumeric):
                                                # constrained numeric type
          retval = fmt.check(value)

        elif issubclass(fmt, Enum):             # enumerated type
          supported_vals = [fm.name for fm in fmt]
          if value not in supported_vals:
            raise ValueError(
              f'Unsupported value. Supported values are {supported_vals}.')
          retval = value

        else:                                   # None of the types matched
          raise ValueError(
            f'The template has an unsupported type: {repr(fmt)}.')

      except Exception as exc:
        exceptions.append(exc)
      else:
        return retval

    if len(exceptions) > 1:
      raise ValueError(
        f'Failed to convert {repr(value)} to any of the desired types. ' +
        'Failed attempts:\n' +
        '\n'.join([f'  {str(exc)}' for exc in exceptions])
      )
    elif len(exceptions) == 1:  # There was only one exception, ...
      raise exceptions[0]       # re-raise it for the caller to handle.


class ConfigError(Exception):
  def __init__(self, file, section=None, field=None, *args, **kwargs):
    self.file = file
    self.section = section
    self.field = field

    super(ConfigError, self).__init__(*args, **kwargs)

  def __str__(self):
    if self.section is not None:
      if self.field is not None:
        loc = f' [{self.section}:{self.field}]'
      else:
        loc = f' [{self.section}]'
    else:
      loc = ''
    return f'({self.file}{loc}): {str(self.args[0])}'


class _ConstrainedNumeric:
  @staticmethod
  @abc.abstractmethod
  def check(val):
    raise NotImplementedError(
      'Programmer error: check() method not implemented in derived class')


class PosInt(_ConstrainedNumeric):
  @staticmethod
  def check(val):
    retval = int(val)
    if retval > 0:
      return retval
    raise ValueError(f'Positive integer expected, got {repr(val)}')


class NonNegInt(_ConstrainedNumeric):
  @staticmethod
  def check(val):
    retval = int(val)
    if retval >= 0:
      return retval
    raise ValueError(f'Non-negative integer expected, got {repr(val)}')


class PosFloat(_ConstrainedNumeric):
  @staticmethod
  def check(val):
    retval = float(val)
    if retval > 0.0:
      return retval
    raise ValueError(f'Positive floating-point value expected, got {repr(val)}')


class NonNegFloat(_ConstrainedNumeric):
  @staticmethod
  def check(val):
    retval = float(val)
    if retval >= 0.0:
      return retval
    raise ValueError(
      f'Non-negative floating-point value expected, got {repr(val)}')


class PosFrac(_ConstrainedNumeric):
  @staticmethod
  def check(val):
    retval = float(val)
    if 0.0 < retval <= 1.0:
      return retval
    raise ValueError(f'Expected positive value <= 1.0, got {repr(val)}')


class NonNegFrac(_ConstrainedNumeric):
  @staticmethod
  def check(val):
    retval = float(val)
    if 0.0 <= retval <= 1.0:
      return retval
    raise ValueError(f'Expected value in the range [0.0-1.0], got {repr(val)}')
