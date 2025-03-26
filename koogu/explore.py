import os
import sys
import numpy as np
import json
import argparse
from functools import partial

from .data import annotations
from .utils import processed_items_generator_mp
from .utils.detections import LabelHelper
from .utils.terminal import ArgparseConverters
from .utils.config import Config, ConfigError
from .utils.filesystem import get_valid_audio_annot_entries


def get_raven_reader_for_explore(**kwargs):
    """
    Returns a Raven.Reader instance that reads rows with 'Spectrogram #' in the
    "View" column. kwargs are passed as-is to the constructor.

    :meta private:
    """

    # Remove if this were specified, since we'll add it below regardless
    if 'fetch_frequencies' in kwargs:
        kwargs.pop('fetch_frequencies')

    return annotations.Raven.Reader(
        fetch_frequencies=True,
        additional_fieldspec=[('View', None, 'Spectrogram 1')],
        filter_fn=_raven_reader_filter_for_explore,
        **kwargs
    )


def _raven_reader_filter_for_explore(selection):
    # Used by the above function
    return selection[0] is not None and \
        selection[1] is not None and \
        selection[-1].startswith('Spectrogram')


def _get_annot_file_vals(filepath, annotation_reader,
                         valid_tags=None,
                         default_bandwidth=None):
    """Read single file"""

    if default_bandwidth is None:
        default_bandwidth = [0, 99999]

    (times, freqs, tags, _, _) = annotation_reader(filepath, multi_file=False)

    if valid_tags is not None:
        valid_mask = [(tag in valid_tags) for tag in tags]

        times = [t for t, v in zip(times, valid_mask) if v]
        if freqs is not None:
            freqs = [f for f, v in zip(freqs, valid_mask) if v]
        tags = [t for t, v in zip(tags, valid_mask) if v]

    # Use default bandwidth if not available
    if freqs is None:
        freqs = [default_bandwidth] * len(times)
    else:
        freqs = [(default_bandwidth[0] if np.isnan(f[0]) else f[0],
                  default_bandwidth[1] if np.isnan(f[1]) else f[1])
                 for f in freqs]

    u_tags, u_idxs = np.unique(tags, return_inverse=True)
    return {
        tag: [
            (times[s_idx][1] - times[s_idx][0],     # duration
             freqs[s_idx][0],                       # low freq
             freqs[s_idx][1])                       # high freq
            for s_idx, u_idx in enumerate(u_idxs)
            if u_idx == t_idx
        ]
        for t_idx, tag in enumerate(u_tags)
    }


def _process_project_annotations(cfg, num_threads=None):

    # if not os.path.exists(cfg.paths.train_annotations):
    #     print('Error: Invalid path specified in train_annotations',
    #           file=sys.stderr)
    #     return 102

    other_args = dict()
    if num_threads is not None:
        other_args['num_threads'] = num_threads

    if cfg.prepare.annotation_reader is not None and \
            cfg.prepare.annotation_reader != 'Raven':
        annotation_reader = getattr(
            annotations, cfg.prepare.annotation_reader).Reader(
                fetch_frequencies=True)
    else:
        # Default to custom raven.Reader
        ar_kwargs = dict()
        if cfg.prepare.raven_label_column_name is not None:
            ar_kwargs['label_column_name'] = \
                cfg.prepare.raven_label_column_name
        if cfg.prepare.raven_default_label is not None:
            ar_kwargs['default_label'] = cfg.prepare.raven_default_label
        annotation_reader = get_raven_reader_for_explore(**ar_kwargs)

    anything_output = False

    if cfg.paths.train_audio_annotations_map is not None:

        if cfg.paths.train_annotations is None or \
                (not os.path.exists(cfg.paths.train_annotations)):
            print('Error: No or invalid path specified in train_annotations',
                  file=sys.stderr)
            return 102

        annot_files = [
            aa[1] for aa in get_valid_audio_annot_entries(
                cfg.paths.train_audio_annotations_map,
                cfg.paths.train_audio, cfg.paths.train_annotations)]

        analyze_annotations(
            cfg.paths.train_annotations, annot_files,
            os.path.join(cfg.paths.project_root,
                         'training_annotations_stats.html'),
            annotation_reader=annotation_reader,
            desired_labels=cfg.prepare.desired_labels,
            remap_labels_dict=cfg.prepare.remap_labels_dict,
            name='Training',
            **other_args
        )

        anything_output = True

    if cfg.paths.test_audio_annotations_map is not None:

        if cfg.paths.test_annotations is None or \
                (not os.path.exists(cfg.paths.test_annotations)):
            print('Error: No or invalid path specified in test_annotations',
                  file=sys.stderr)
            return 102

        annot_files = [
            aa[1] for aa in get_valid_audio_annot_entries(
                cfg.paths.test_audio_annotations_map,
                cfg.paths.test_audio, cfg.paths.test_annotations)]

        analyze_annotations(
            cfg.paths.test_annotations, annot_files,
            os.path.join(cfg.paths.project_root, 'test_annotations_stats.html'),
            annotation_reader=annotation_reader,
            desired_labels=cfg.prepare.desired_labels,
            remap_labels_dict=cfg.prepare.remap_labels_dict,
            name='Test',
            **other_args
        )

        anything_output = True

    if not anything_output:
        print('Nothing processed. Please check project config.')

    return 0    # All succeeded


def _process_non_project_annotations(annots_root,
                                     annotation_reader=None,
                                     filelist_file=None,
                                     output_file=None,
                                     num_threads=None,
                                     **kwargs):

    other_args = dict()
    if num_threads is not None:
        other_args['num_threads'] = num_threads

    if annotation_reader is not None and annotation_reader != 'Raven':
        annotation_reader = getattr(
            annotations, annotation_reader).Reader(fetch_frequencies=True)
    else:
        # Default to Raven.Reader
        ar_kwargs = dict()
        if 'label_column_name' in kwargs:
            ar_kwargs['label_column_name'] = kwargs.get('label_column_name')
        if 'default_label' in kwargs:
            ar_kwargs['default_label'] = kwargs.get('default_label')
        annotation_reader = get_raven_reader_for_explore(**ar_kwargs)

    if filelist_file is not None:
        # Gather all non-blank lines
        with open(filelist_file, 'r') as in_f:
            filelist = [line.strip() for line in in_f if len(line) > 1]

        # Check files' existence (& complain)
        valid_mask = [
            os.path.exists(f2) and os.path.isfile(f2)
            for f2 in (os.path.join(annots_root, f1) for f1 in filelist)
        ]
        if not all(valid_mask):
            print('Following entries are invalid and will be ignored:')
            for f1, m in zip(filelist, valid_mask):
                if not m:
                    print(f'  {f1}')

        filelist = [f1 for f1, m in zip(filelist, valid_mask) if m]

    else:
        extn = getattr(
            annotations, annotation_reader.__module__.split('.')[-1]
        ).default_extension()
        filelist = [
            f1 for f1 in os.listdir(annots_root)
            if f1.endswith(extn) and os.path.isfile(
                os.path.join(annots_root, f1))
        ]

    analyze_annotations(
        annots_root, filelist,
        output_file or 'annotations_stats.html',
        annotation_reader=annotation_reader,
        **other_args
    )

    return 0


def analyze_annotations(annot_root, annot_files, outfile,
                        annotation_reader=None,
                        desired_labels=None,
                        remap_labels_dict=None,
                        default_bandwidth=None,
                        name=None,
                        **kwargs):

    if annotation_reader is None:
        annotation_reader = annotations.Raven.Reader(fetch_frequencies=True)

    # Gather (duration, low freq, high freq) from each file
    tag_vals = dict()
    for _, file_tag_vals in processed_items_generator_mp(
            kwargs.pop('num_threads', os.cpu_count() or 1),
            _get_annot_file_vals,
            (os.path.join(annot_root, anf) for anf in annot_files),
            annotation_reader,
            default_bandwidth=default_bandwidth):

        for tag, vals in file_tag_vals.items():
            if tag in tag_vals:
                tag_vals[tag] += vals
            else:
                tag_vals[tag] = vals

    if desired_labels is not None:
        is_fixed_classes = True
        classes_list = desired_labels
    else:
        # Use all available classes
        is_fixed_classes = False
        classes_list = sorted([k for k in tag_vals.keys()])

    label_helper = LabelHelper(
        classes_list,
        remap_labels_dict=remap_labels_dict,
        fixed_labels=is_fixed_classes)

    final_stats = [[] for _ in label_helper.classes_list]
    for old_lbl, idx in label_helper.labels_to_indices.items():
        if old_lbl in tag_vals:
            final_stats[idx] += [       # also include bandwidth now
                (du, lf, hf, hf - lf)
                for (du, lf, hf) in tag_vals[old_lbl]
            ]

    del tag_vals

    # Write header and start body
    with open(outfile, 'w', encoding='utf-8') as of:
        of.write("""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Koogu - Analyze Annotations</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style type="text/css">
    div.container {
      overflow-x: scroll;
      display: flex;
    }
    div.content {
    }
    div.sidebar {
      width:300px;
      box-shadow: -5px 0 5px rgba(22,22,22,.5);
      border-left: darkgray solid 1px;
      display: flex;
      flex-flow: column;
      flex-shrink: 0;
      overflow-y: scroll;
      padding-left: 5px;
      font-family: "Open Sans", verdana, arial, sans-serif;
    }
    div.sidebar-header {
        margin-top: 4px;
        margin-bottom: 2px;
        font-weight: bold;
    }
    div.tHead {
      margin: 0 5px 8px 5px;
      height: 2.2em;
    }
    div.tCont {
      transition: box-shadow .05s;
      border: 1px solid #ccc;
      border-radius: 10px 10px 5px 5px;
      margin: 4px 6px 8px 5px;
    }
    div.tCont:hover {
      box-shadow: 3px 4px 6px rgba(22,22,44,.5);
      margin-top: -1px;
      margin-left: 4px;
    }
    div.tCont:hover > div.tName {
      background: #aaa;
    }
    div.tName {
      padding: 4px 5px;
      background: #ccc;
      border-radius: 10px 10px 0 0;
    }
    div.tItemsCont {
      display: flex;
      flex-wrap: nowrap;
      font-family: Arial, Helvetica, sans-serif;
    }
    .bottomLine {
      border-bottom: gray 1px dotted;
    }
    div.tItemsCont > div {
      padding: 0 5px;
    }
    div.tItemName {
      width: 19ch;
      min-width: 19ch;
    }
    div.tableCell {
      display: flex;
      flex-flow: row nowrap;
      margin-top: auto;
    }
    div.tMetrRange {
      width: 20ch;
      min-width: 20ch;
    }
    div.tMetr {
      width: 48ch;
      min-width: 48ch;
    }
    div.tableCell > div.left {
      text-align: right;
      padding: 0 4px 0 0;
      width: 50%;
    }
    div.tableCell > div.right {
      text-align: left;
      padding: 0 0 0 4px;
      width: 50%;
    }
    div.headerCell {
      flex-direction: column;
      text-align: center;
      margin-top: auto;
    }
    .tLabel {
      font-size: 1.05em;
      font-style: italic;
      font-weight: bold;
      font-family: "Times New Roman", Times, serif;
    }
    .tItemNumeric {
      font-size: 0.9em;
      font-family: "Lucida Console", "Courier New", monospace;
    }
    div.freqPlots {
      display: flex;
      flex-flow: row nowrap;
      padding-bottom: 10px;
    }
    div.freqPlotsL {
      width: 60%;
      padding-right: 5px;
    }
    div.freqPlotsR {
      width: 40%;
      padding-left: 5px;
    }
    div.durPlots {
    }
    #divPlotly {
        width: 100%;
        height: 100%;
    }
    .orspec {
      height: 320px;
    }
  </style>
</head>
<body style="margin: 0;">

  <div style="display: flex; flex-flow: row nowrap; height: 100vh;">
    <div class="container">
      <div class="content">
        <div class="tHead">
          <div class="tItemsCont">
            <div class="tItemName headerCell">
              <div class="tableCell"></div>
            </div>
            
            <div class="headerCell tMetrRange tItemNumeric">
              <div class="tableCell">
                <div class="left">Min.</div>
                <div>-</div>
                <div class="right">Max.</div>
              </div>
            </div>
            
            <div class="headerCell tMetrRange tItemNumeric">
              <div class="tableCell">
                <div class="left">Mean</div>
                <div>±</div>
                <div class="right">Std.</div>
              </div>
            </div>
            
            <div class="headerCell tMetr tItemNumeric">
              <div class="tableCell bottomLine"
                style="justify-content: center;">Percentiles</div>
              <div class="tableCell">
                <div class="left" style="width: 20%;">5th</div>
                <div class="left" style="width: 20%;">25th</div>
                <div class="left" style="width: 20%;">50th</div>
                <div class="left" style="width: 20%;">75th</div>
                <div class="left" style="width: 20%;">95th</div>
              </div>
            </div>
          </div>
        </div>
        """)

        field_names = ['Duration (s)', 'Low Frequency (Hz)',
                       'High Frequency (Hz)', 'Bandwidth (Hz)']
        field_colors = ['#5da4d6', '#2ca065', '#ff900e', '#cf72ff']

        for tag_idx, pt, pd in zip(range(len(label_helper.classes_list)),
                                   label_helper.classes_list,
                                   final_stats):

            of.write(
                f'      <div class="tCont" onmouseover=\'updateTagInfo({json.dumps(pt)}, {tag_idx});\'>\n')  # tag-specific container
            of.write(
                f'        <div class="tName"><span class="tLabel">{pt}</span>: {len(pd)} annotations </div>\n')

            minvals = maxvals = meanvals = stdvals = \
                [np.nan] * len(field_names)
            prcsvals = [[np.nan] * 5] * len(field_names)
            if len(pd) > 0:
                minvals = np.min(pd, axis=0)
                maxvals = np.max(pd, axis=0)
                if len(pd) > 2:
                    meanvals = np.mean(pd, axis=0)
                    stdvals = np.std(pd, axis=0)
                    prcsvals = np.percentile(pd, [5, 25, 50, 75, 95], axis=0).T
            for metr_idx, (c, min, max, mean, std, prcs) in enumerate(
                    zip(field_names,
                        minvals, maxvals, meanvals, stdvals, prcsvals)
                    ):

                of.write('        <div class="tItemsCont{}">'.format(
                    ' bottomLine' if metr_idx < len(field_names) - 1 else ''))
                of.write(f'<div class="tableCell tItemName" style="color: {field_colors[metr_idx]};">{c:<3s}</div>\n')
                of.write('<div class="tableCell tMetrRange tItemNumeric">\n')
                of.write(
                    f'<div class="left">{min:.2f}</div><div>-</div><div class="right">{max:.2f}</div>\n')
                of.write('</div>\n')
                of.write('<div class="tableCell tMetrRange tItemNumeric">\n')
                of.write(
                    f'<div class="left">{mean:.2f}</div><div>±</div><div class="right">{std:.2f}</div>\n')
                of.write('</div>\n')
                of.write('<div class="tableCell tMetr tItemNumeric">\n')
                of.write(''.join(
                    [f'<div class="left" style="width: 20%;">{prc:.2f}</div>\n'
                     for prc in prcs]))
                of.write('</div>\n')
                of.write('        </div>')
            of.write('      </div>\n')

        of.write('    </div>\n')
        of.write('  </div>\n')

        of.write("""  <div class="sidebar">
    <div class="sidebar-header">Tag</div>
    <div id="hv_tg">&nbsp;</div>
    <div class="sidebar-header">&nbsp;</div>
    <div>
      <div class="freqPlots">
        <div class="freqPlotsL"><div id="freqsPlot"></div></div>
        <div class="freqPlotsR"><div id="bwPlot"></div></div>
      </div>
      <div class="durPlots">
        <div id="durPlot"></div>
      </div>
    </div>
  </div>
        """)

        of.write('</div>')

        of.write('<script>\n')
        of.write("""
  const layout = {
    paper_bgcolor: 'rgb(245,245,245)',
    plot_bgcolor: 'rgb(250,250,250)',
    hovermode: false,
    showlegend: false,
    boxgap: 0.6
  };
  const plotCfg = {
    modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
    displaylogo: false
  };

  Plotly.newPlot("freqsPlot", [{y: [], type: 'box', name: 'Low Frequency', jitter: 0.5, marker: {size: 3, color: 'rgb(44, 160, 101)'}, line: {width: 1}, notched: true}, {y: [], type: 'box', name: 'High Frequency', jitter: 0.5, marker: {size: 3, color: 'rgb(255, 144, 14)'}, line: {width: 1}, notched: true}],
    {...layout, height: 250, margin: {l: 50, r: 0, b: 0, t: 0}, xaxis: {showgrid: false, zeroline: false, showticklabels: false, ticks: ""}, yaxis: {automargin: true, zeroline: false, title: "Frequency (Hz)"}},
    plotCfg);
  Plotly.newPlot("bwPlot", [{y: [], type: 'box', name: 'Bandwidth', jitter: 0.5, marker: {size: 3, color: 'rgb(207, 114, 255)'}, line: {width: 1}, notched: true}],
    {...layout, height: 250, margin: {l: 55, r: 0, b: 0, t: 0}, xaxis: {showgrid: false, zeroline: false, showticklabels: false, ticks: ""}, yaxis: {automargin: true, zeroline: false, title: "Frequency (Hz)"}},
    plotCfg);
  Plotly.newPlot("durPlot", [{x: [], type: 'box', name: 'Duration', orientation: 'h', jitter: 0.5, marker: {size: 3, color: 'rgb(93, 164, 214)'}, line: {width: 1}, notched: true}],
    {...layout, height: 100, margin: {l: 0, r: 0, b: 35, t: 0}, yaxis: {showgrid: false, zeroline: false, showticklabels: false, ticks: ""}, xaxis: {automargin: true, zeroline: false, title: "Time (s)"}},
    plotCfg);


  const stats = [
        """)
        for tag_idx, pd in enumerate(final_stats):
            of.write('    [\n')
            for metr_idx in range(4):
                of.write('      [{:s}],\n'.format(
                    ','.join([f'{v[metr_idx]}' for v in pd])))
            of.write('    ],\n')
        of.write('  ];\n')
        of.write("""  
  function updateTagInfo(tagLabel, tagIdx) {
    var labelField = document.getElementById('hv_tg');
    labelField.innerHTML = tagLabel;

    Plotly.restyle("freqsPlot", {y: [stats[tagIdx][1], stats[tagIdx][2]]}, [0, 1]);
    Plotly.restyle("bwPlot", {y: [stats[tagIdx][3]]});
    Plotly.restyle("durPlot", {x: [stats[tagIdx][0]]});
  };

        """)
        of.write('</script>\n')

        of.write('\n'.join([
            '</body>',
            '</html>\n']))

    print((f'{name} a' if name else 'A') +
          f'nnotations\' analysis written to\n  {outfile}')


__all__ = []


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu.explore', allow_abbrev=True,
            description='Explore available training & validation datasets.')

    parser.add_argument(
        'cfg_file', metavar='<CONFIG FILE/ANNOTATIONS DIR>',
        help='If running within the scope a Koogu project, specify path to the '
             'project config file (training and test annotations\' info will be'
             ' read from the specified config file). Otherwise, specify path to'
             ' the directory containing the set of annotation files you wish to'
             ' analyze (also see additional options under `Direct access`).'
    )

    arg_group_direct_ctrl = parser.add_argument_group(
        'Direct access',
        description='These options are only applicable when running outside '
                    'the scope of a Koogu project, i.e., directly accessing '
                    'annotation files by specifying <ANNOTATIONS DIR>. If '
                    '<CONFIG FILE> was specified, all these below options will '
                    'be ignored.')
    arg_group_direct_ctrl.add_argument(
        '--reader', dest='annotation_reader',
        choices=annotations.__all__, default=annotations.__all__[0],
        help='Set this based on the format of your annotation files.')
    arg_group_direct_ctrl.add_argument(
        '--filelist', dest='filelist_file', metavar='FILE',
        help='Path to a text file containing one-per-line entries of the '
             'annotation files to analyze. If not specified, all discoverable '
             'files under <ANNOTATIONS DIR> will be analyzed.')
    arg_group_direct_ctrl.add_argument(
        '--output', dest='output_file', metavar='FILE',
        help='Analysis outputs (HTML) will be written to this file, if '
             'specified. Otherwise, outputs will be written to the file '
             '`annotation_stats.html` in the current directory.')

    arg_group_prctrl = parser.add_argument_group('Process control')
    arg_group_prctrl.add_argument(
        '--threads', dest='num_threads', metavar='NUM',
        type=ArgparseConverters.positive_integer,
        help='Number of threads to spawn for parallel execution (default: as ' +
             'many CPUs).')

    # arg_group_logging = parser.add_argument_group('Logging')
    # arg_group_logging.add_argument(
    #     '--loglevel', dest='log_level',
    #     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    #     default='INFO', help='Logging level.')

    parser.set_defaults(exec_fn=cmdline_run)

    return parser


def cmdline_run(cfg_file, num_threads=None,
                annotation_reader=None, filelist_file=None, output_file=None,
                # Other overriding parameters not available via cmdline
                ):
    """Functionality invoked via the command-line interface"""

    # Load config
    try:
        cfg = Config(cfg_file, 'prepare')
    except FileNotFoundError as exc:
        print(f'Error loading config file: {exc.strerror}', file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print(f'Error processing config file: {str(exc)}', file=sys.stderr)
        exit(1)
    except IsADirectoryError as _:
        # Analyze non-project annotations.
        func = partial(_process_non_project_annotations,
                       annots_root=cfg_file,
                       filelist_file=filelist_file,
                       annotation_reader=annotation_reader,
                       output_file=output_file)
    except Exception as exc:
        print(f'Error processing config file: {repr(exc)}', file=sys.stderr)
        exit(1)
    else:
        func = partial(_process_project_annotations, cfg=cfg)

    try:
        retval = func(num_threads=num_threads)
    except (FileNotFoundError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        retval = exc.errno

    exit(retval)

