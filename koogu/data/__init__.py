
import os
import h5py

from koogu.data.raw import Settings, Audio, Convert, Process


class FilenameExtensions:
    """Extensions for common file types.

    The following extensions are defined:

    * `numpy`: '.npz'
    * `hdf5`: '.hdf5'
    * `tfrecord`: '.tfrecord'
    * `json`: '.json'
    """

    numpy = '.npz'
    hdf5 = '.hdf5'
    tfrecord = '.tfrecord'
    json = '.json'


class DirectoryNames:
    """Names of common directory types.

    The following directory names are defined:

    * `TRAIN`: 'train'
    * `EVAL`: 'val'
    * `TEST`: 'test'
    """

    TRAIN = 'train'
    EVAL = 'val'
    TEST = 'test'


class AssetsExtraNames:
    """Names of fields in assets.extra, that is common between training and inference phases of operation."""

    classes_list = 'classes_list' + FilenameExtensions.json
    audio_settings = 'audio_settings' + FilenameExtensions.json
    spec_settings = 'spec_settings' + FilenameExtensions.json


class DatasetDigest:
    """Abstract class for handling (creating, altering, reading) digest information pertaining to TFRecord datasets.
    Note that each call of the class' methods access the digest file. Maintaining consistency across related data
    entries within the file are up to the caller; no validation is done in here."""

    # Names starting with 's_' indicate HDF5 datasets and with 'g_' indicate HDF5 Groups
    _classes_list = 's_classes'
    _classes_group = 'g_classes'
    _data_shape = 's_data_shape'
    _per_class_per_group_spec_counts = 's_specs_per_class_and_group'

    @staticmethod
    def _get_filepath(record_dir):
        return os.path.join(record_dir, 'dataset_digest' + FilenameExtensions.hdf5)

    @staticmethod
    def Wipeout(record_dir):
        """Simply deletes the digest file. Use with caution."""
        digest_file = DatasetDigest._get_filepath(record_dir)
        if os.path.exists(digest_file):
            os.unlink(digest_file)

    @staticmethod
    def AddOrderedClassList(record_dir, class_list):
        digest_file = DatasetDigest._get_filepath(record_dir)

        with h5py.File(digest_file, 'a') as hf:
            hf.create_dataset(DatasetDigest._classes_list, (len(class_list), ), h5py.special_dtype(vlen=bytes),
                              [class_name.encode('unicode_escape') for class_name in class_list])

    @staticmethod
    def GetOrderedClassList(record_dir):
        digest_file = DatasetDigest._get_filepath(record_dir)

        if os.path.exists(digest_file):
            with h5py.File(digest_file, 'r') as hf:
                if DatasetDigest._classes_list in hf:
                    return [class_name.decode('unicode_escape') for class_name in hf[DatasetDigest._classes_list][()]]

        return None     # If any of the above conditions failed

    @staticmethod
    def GetNumClasses(record_dir):
        digest_file = DatasetDigest._get_filepath(record_dir)

        if os.path.exists(digest_file):
            with h5py.File(digest_file, 'r') as hf:
                if DatasetDigest._classes_list in hf:
                    return len(hf[DatasetDigest._classes_list][()])

        return None     # If any of the above conditions failed

    @staticmethod
    def AddClassOrderedSpecFileList(record_dir, class_name, spec_files):
        # Add to the digest file
        digest_file = DatasetDigest._get_filepath(record_dir)

        with h5py.File(digest_file, 'a') as hf:
            if DatasetDigest._classes_group in hf:
                classes_grp = hf[DatasetDigest._classes_group]
            else:
                classes_grp = hf.create_group(DatasetDigest._classes_group)

            classes_grp.create_dataset(
                class_name.encode('unicode_escape'), (len(spec_files), ), h5py.special_dtype(vlen=bytes),
                [s_file.encode('unicode_escape') for s_file in spec_files])

    @staticmethod
    def GetClassOrderedSpecFileList(record_dir, class_name):
        # Add to the digest file
        digest_file = DatasetDigest._get_filepath(record_dir)

        try:
            with h5py.File(digest_file, 'r') as hf:
                return [s_file.decode('unicode_escape') for s_file in
                        hf[(DatasetDigest._classes_group + '/' + class_name).encode('unicode_escape')]]
        except Exception as _:
            return None

    @staticmethod
    def AddDataShape(record_dir, data_shape):
        digest_file = DatasetDigest._get_filepath(record_dir)

        with h5py.File(digest_file, 'a') as hf:
            hf.create_dataset(DatasetDigest._data_shape, data=data_shape, dtype='u4')

    @staticmethod
    def GetDataShape(record_dir):
        digest_file = DatasetDigest._get_filepath(record_dir)

        try:
            with h5py.File(digest_file, 'r') as hf:
                return hf[DatasetDigest._data_shape][()]
        except Exception as _:
            return None

    @staticmethod
    def AddPerClassAndGroupSpecCounts(record_dir, specs_per_class_and_group):
        digest_file = DatasetDigest._get_filepath(record_dir)

        with h5py.File(digest_file, 'a') as hf:
            hf.create_dataset(DatasetDigest._per_class_per_group_spec_counts, data=specs_per_class_and_group,
                              dtype='u8')

    @staticmethod
    def GetPerClassAndGroupSpecCounts(record_dir):
        digest_file = DatasetDigest._get_filepath(record_dir)

        try:
            with h5py.File(digest_file, 'r') as hf:
                return hf[DatasetDigest._per_class_per_group_spec_counts][()]
        except Exception as _:
            return None

    @staticmethod
    def GetTrainingSpecCounts(record_dir):
        counts = DatasetDigest.GetPerClassAndGroupSpecCounts(record_dir)

        if counts is not None:
            counts = counts[:, 0].sum()
        return counts

    @staticmethod
    def GetValidationSpecCounts(record_dir):
        counts = DatasetDigest.GetPerClassAndGroupSpecCounts(record_dir)

        if counts is not None:
            counts = counts[:, 1].sum()
        return counts

    @staticmethod
    def GetTestingSpecCounts(record_dir):
        counts = DatasetDigest.GetPerClassAndGroupSpecCounts(record_dir)

        if counts is not None:
            counts = counts[:, 2].sum()
        return counts


__all__ = [Settings, Audio, Convert, Process, DatasetDigest]
