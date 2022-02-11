
class FilenameExtensions:
    """Extensions for common file types.

    The following extensions are defined:

    * `numpy`: '.npz'
    * `json`: '.json'
    """

    numpy = '.npz'
    json = '.json'


class AssetsExtraNames:
    """
    Names of fields in assets.extra, that is common between training and
    inference phases of operation.
    """

    classes_list = 'classes_list' + FilenameExtensions.json
    audio_settings = 'audio_settings' + FilenameExtensions.json
    spec_settings = 'spec_settings' + FilenameExtensions.json
