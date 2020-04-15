
import abc
import tensorflow as tf


class TFRecordHandler(metaclass=abc.ABCMeta):
    """
    Base class to handle functionality of reading from and writing to TFRecord files.
    Inherited class must redefine x_name as a class-level attribute.
    """

    @property
    @abc.abstractmethod
    def x_name(self):   # Field name. Derived class must implement this as a property
        pass

    # Field names for Y values
    y_name = 'class'
    y_name1 = 'file_idx'    # File id. Used only if tracking info is turned on
    y_name2 = 'samp_idx'    # Sample id. Used only if tracking info is turned on

    # Configuration for parsing each field item in a record
    _default_field_configurations = {
        'x': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        y_name: tf.io.FixedLenFeature([], tf.int64),
        y_name1: tf.io.FixedLenFeature([], tf.int64),
        y_name2: tf.io.FixedLenFeature([], tf.int64)
    }

    # Currently supported datatypes
    _type_to_feature_fn_map = {
        tf.float32: lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x)),
        tf.int64: lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x if hasattr(x, '__len__') else [x]))
        # tf.int8: tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    }

    def __init__(self, tracing_on=False, field_configurations=None):
        """ ...
        Handles 4 fields, three of which are 'class', 'file_idx' and 'samp_idx' (sample index), and the 4th field must
        be defined in an inherited class by overriding the x_name property.

        :param tracing_on: (bool, default False) Whether or not to store spec tracing info in TFRecords.
        :param field_configurations: If not None, must be a dict containing a mapping from a field name to a TensorFlow
            configuration object (such as tf.io.FixedLenFeature) that will override the default configuration for a
            field. E.g.   field_configurations = { 'class': tf.FixedLenFeature([], tf.int64) }
            will override default configuration of just the 'class' field.

        """

        if field_configurations is None:
            field_configurations = {}
        else:
            # Ensure that only those configuration types are provided that can be currently handled
            assert all([(t in self._type_to_feature_fn_map.keys()) for t in
                        [f.dtype for _, f in field_configurations.items()]])

        self._keys_to_features = {
            # Feature/data
            self.x_name: field_configurations[self.x_name] if self.x_name in field_configurations.keys() \
                else self._default_field_configurations['x'],

            # Label
            self.y_name: field_configurations[self.y_name] if self.y_name in field_configurations.keys() \
                else self._default_field_configurations[self.y_name]
        }

        if tracing_on:      # Also include file ID and sample ID as tracing info

            # File ID
            self._keys_to_features[self.y_name1] = \
                field_configurations[self.y_name1] if self.y_name1 in field_configurations.keys() \
                else self._default_field_configurations[self.y_name1]

            # Sample ID
            self._keys_to_features[self.y_name2] = \
                field_configurations[self.y_name2] if self.y_name2 in field_configurations.keys() \
                else self._default_field_configurations[self.y_name2]

        self._features_to_keys_types = {f: c.dtype for f, c in self._keys_to_features.items()}

    def write_record(self, writer, feature_data, feature_label, file_idx=0, sample_idx=0):
        """
        Use this method for writing data to TFRecord files.

        :param writer: A valid TFRecordWriter object
        :param feature_data: ...
        :param feature_label: ...
        :param file_idx: ...
        :param sample_idx: ...
        :return: NA
        """

        temp = {
            self.x_name: feature_data.ravel(),
            self.y_name: feature_label,
            self.y_name1: file_idx,
            self.y_name2: sample_idx}

        # Keep only whatever fields exist in _features_to_keys_type
        feature_dict = {k: self._type_to_feature_fn_map[t](temp[k])
                        for k, t in self._features_to_keys_types.items()}

        tf_features = tf.train.Features(feature=feature_dict)
        example = tf.train.Example(features=tf_features)
        writer.write(example.SerializeToString())

    def parse_record(self, record):
        """Use this method in training/validation input pipelines for reading from TFRecord files"""

        features = tf.io.parse_single_example(record, features=self._keys_to_features)

        if self.y_name1 in self._keys_to_features.keys():  # and self.y_name2 in self._keys_to_features.keys():
            return features[self.x_name], features[self.y_name], features[self.y_name1], features[self.y_name2]
        else:
            return features[self.x_name], features[self.y_name]


class WaveformTFRecordHandler(TFRecordHandler):

    x_name = 'clip'     # Override name of the feature field

    def __init__(self, tracing_on=False, **kwargs):

        # Invoke the parent constructor
        super(WaveformTFRecordHandler, self).__init__(tracing_on=tracing_on, **kwargs)


class SpectrogramTFRecordHandler(TFRecordHandler):

    x_name = 'spec'     # Override name of the feature field

    def __init__(self, tracing_on=False, **kwargs):

        # Invoke the parent constructor
        super(SpectrogramTFRecordHandler, self).__init__(tracing_on=tracing_on, **kwargs)

