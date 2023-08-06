import os
import numpy as np
from koogu.data import annotations
import pytest


@pytest.mark.parametrize(
    'reader_type, file, attempt_fetch_freq, freqs_must_exist, files_must_exist',
    [
        (annotations.Raven, 'raven_no_freq.txt', False, True, False),
        (annotations.Raven, 'raven_no_freq.txt', True, False, False),
        (annotations.Raven, 'raven_with_freq.txt', False, True, False),
        (annotations.Raven, 'raven_with_freq.txt', True, True, False),
        (annotations.Raven, 'raven_with_freq_multifile.txt', True, True, True),
        (annotations.Audacity, 'audacity_no_freq.txt', False, True, False),
        (annotations.Audacity, 'audacity_no_freq.txt', True, False, False),
        (annotations.Audacity, 'audacity_with_freq.txt', False, True, False),
        (annotations.Audacity, 'audacity_with_freq.txt', True, True, False)
    ]
)
def test_reader_basics(dataroot,
                       reader_type, file,
                       attempt_fetch_freq, freqs_must_exist, files_must_exist):

    reader = reader_type.Reader(fetch_frequencies=attempt_fetch_freq)

    (times, freqs, tags, chs, files) = reader(
        os.path.join(dataroot, 'annotation_formats', file),
        multi_file=files_must_exist
    )

    expected_times = [(1.0, 1.0),
                      (2.0, 2.5),
                      (2.5, 2.75),
                      (3.0, 3.5),
                      (4.0, 4.5)]
    expected_freqs = [(6928.456055, 6928.456055),
                      (np.nan, 6334.588867),
                      (np.nan, np.nan),
                      (8908.015625, np.nan),
                      (1484.669312, 2969.338523)]
    expected_chs = [0, 0, 0, 0, 0]
    expected_files = ['a', 'b', 'c', 'd', 'e']

    # Compare times
    assert all([(t1[0] == t2[0] and t1[1] == t2[1])
                for t1, t2 in zip(times, expected_times)]), times

    # Compare frequencies, conditionally
    if attempt_fetch_freq:
        assert freqs is not None, 'Expected frequencies, got None'

        if freqs_must_exist:
            assert all([
                ((f1[0] == f2[0] or (np.isnan(f1[0]) and np.isnan(f2[0]))) and
                 (f1[1] == f2[1]) or (np.isnan(f1[1]) and np.isnan(f2[1])))
                for f1, f2 in zip(freqs, expected_freqs)])
        else:
            assert all([np.isnan(v) for row in freqs for v in row]), \
                f'Expected all NaNs, got {freqs}'

    else:
        assert freqs is None, f'Was NOT expecting frequencies, got {freqs}'

    # Check, conditionally, if per-annotation source files are available
    if files_must_exist:
        assert files == expected_files

    else:
        assert files is None, f'Was NOT expecting files, got {files}'

    # Check and compare channels
    assert isinstance(chs[0], int)
    assert all([ch1 == ch2
                for ch1, ch2 in zip(chs, expected_chs)])


def test_Raven_Reader_filtering(dataroot):

    annot_file = 'NOPP6_20090329_RW_upcalls.selections.txt'

    reader = annotations.Raven.Reader(
        label_column_name='Tags')

    filtering_reader = annotations.Raven.Reader(
        label_column_name='Tags',
        additional_fieldspec=[('Notes', None)],
        filter_fn=lambda sel: sel[-1] != '?'
    )

    times, _, _, _, _ = reader(
        os.path.join(dataroot, 'narw_dclde', 'train_annotations', annot_file))
    unfiltered_count = len(times)
    assert unfiltered_count == 2279

    times_filtered, _, _, _, _ = filtering_reader(
        os.path.join(dataroot, 'narw_dclde', 'train_annotations', annot_file))
    filtered_count = len(times_filtered)
    assert filtered_count == 2160
