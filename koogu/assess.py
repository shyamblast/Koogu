import os
import sys
import argparse
import warnings
import json
from .data import annotations
from .utils.config import Config, ConfigError
from .utils.assessments import PrecisionRecall


__all__ = []


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu.assess', allow_abbrev=False,
            description='Assess recognition performance of trained model.')

    parser.add_argument(
        'cfg_file', metavar='<CONFIG FILE>',
        help='Path to config file.')

    parser.add_argument(
        'result_file', metavar='<RESULT FILE>',
        help='Path to output file where results are to be written. Results can '
             'be generated in one of the following formats: csv, json, html. '
             'Format is inferred from <RESULT FILE>, and where that\'s not '
             'possible, defaults to csv format.')

    parser.add_argument(
        '--assess-raw', dest='assess_raw', action='store_true',
        help='If set, will assess "raw" clip-level recognition performance. By '
             'default, will assess performance from post-processed detections.')

    # TODO: Add include_reject_class_perf to the above group

    parser.set_defaults(exec_fn=cmdline_assess_performance)

    return parser


def cmdline_assess_performance(cfg_file, result_file, assess_raw):
    """Functionality invoked via the command-line interface"""

    # Load config
    try:
        cfg = Config(cfg_file, 'data.annotations', 'assess', 'prepare')
    except FileNotFoundError as exc:
        print(f'Error loading config file: {exc.strerror}', file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print(f'Error processing config file: {str(exc)}', file=sys.stderr)
        exit(1)
    except Exception as exc:
        print(f'Error processing config file: {repr(exc)}', file=sys.stderr)
        exit(1)

    other_pr_kwargs = dict()
    other_pr_kwargs['thresholds'] = \
        cfg.assess.thresholds if cfg.assess.thresholds is not None else \
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]     # Default
    if cfg.prepare.negative_class is not None:
        other_pr_kwargs['reject_classes'] = cfg.prepare.negative_class

    # Set up annotation reader for reading test annotations
    if cfg.data.annotations.annotation_reader is not None:
        ar_type = getattr(annotations,
                          cfg.data.annotations.annotation_reader).Reader
    else:
        ar_type = annotations.Raven.Reader  # Default to Raven.Reader
    ar_kwargs = dict()
    if ar_type == annotations.Raven.Reader:
        if cfg.data.annotations.raven_label_column_name is not None:
            ar_kwargs['label_column_name'] = \
                cfg.data.annotations.raven_label_column_name
        if cfg.data.annotations.raven_default_label is not None:
            ar_kwargs['default_label'] = \
                cfg.data.annotations.raven_default_label
    other_pr_kwargs['annotation_reader'] = ar_type(**ar_kwargs)
    other_pr_kwargs['remap_labels_dict'] = \
        cfg.data.annotations.remap_labels_dict

    if assess_raw:  # "raw" performance assessments if requested
        if cfg.prepare.min_annotation_overlap_fraction is not None:
            other_pr_kwargs['min_annotation_overlap_fraction'] = \
                cfg.prepare.min_annotation_overlap_fraction
        if cfg.prepare.max_nonmatch_overlap_fraction is not None:
            other_pr_kwargs['max_nonmatch_overlap_fraction'] = \
                cfg.prepare.max_nonmatch_overlap_fraction
    else:
        other_pr_kwargs['post_process_detections'] = True

        if cfg.assess.min_gt_coverage is not None:
            other_pr_kwargs['min_gt_coverage'] = cfg.assess.min_gt_coverage
        if cfg.assess.min_det_usage is not None:
            other_pr_kwargs['min_det_usage'] = cfg.assess.min_det_usage
        if cfg.assess.squeeze_min_dur is not None:
            other_pr_kwargs['squeeze_min_dur'] = cfg.assess.squeeze_min_dur

    # Check output file & format
    rf_ext = os.path.splitext(result_file)[1]
    if rf_ext not in ['.csv', '.json', '.html', '.htm']:
        result_type = 'csv'
        warnings.warn(
            f'Extension of <RESULT FILE> ({rf_ext[1:]}) not among permitted '
            'types. Will generate outputs in the default csv format.')
    else:
        result_type = rf_ext[1:]

    # Assess
    per_class_pr, overall_pr = PrecisionRecall(
        cfg.paths.test_audio_annotations_map,
        cfg.paths.test_detections,
        cfg.paths.test_annotations,
        **other_pr_kwargs
    ).assess()

    os.makedirs(os.path.split(result_file)[0], exist_ok=True)
    if result_type == 'csv':
        _save_csv(result_file, other_pr_kwargs['thresholds'],
                  overall_pr, per_class_pr)
    elif result_type == 'json':
        _save_json(result_file, other_pr_kwargs['thresholds'],
                   overall_pr, per_class_pr)
    elif result_type in ['html', 'htm']:
        _save_html(result_file, other_pr_kwargs['thresholds'],
                   overall_pr, per_class_pr)


def _save_json(result_file, thresholds, overall_pr, per_class_pr):

    with open(result_file, 'w') as outf:
        json.dump(
            dict(
                thresholds=thresholds,
                overall_pr={
                    k: v.tolist() for k, v in overall_pr.items()},
                per_class_pr={
                    cl: {k: v.tolist() for k, v in pr.items()}
                    for cl, pr in per_class_pr.items()
                }

            ),
            outf, indent=2)


def _save_csv(result_file, thresholds, overall_pr, per_class_pr):

    classes = list(per_class_pr.keys())

    with open(result_file, 'w') as outf:

        # Header
        outf.write('{:s}\n'.format(','.join(
            ['Threshold', 'Overall Precision', 'Overall Recall'] +
            [f'{cl} ({ty})' for cl in classes for ty in ['Precision', 'Recall']]
        )))

        types = ['precision', 'recall']
        for th_idx, thld in enumerate(thresholds):
            row = [thld] + \
                  [overall_pr[ty][th_idx] for ty in types] + \
                  [per_class_pr[cl][ty][th_idx]
                   for cl in classes for ty in types]
            outf.write('{:s}\n'.format(','.join(map(str, row))))


def _save_html(result_file, thresholds, overall_pr, per_class_pr):

    outfname = os.path.splitext(os.path.split(result_file)[-1])[0]

    with open(result_file, 'w', encoding='utf-8') as outf:
        outf.write("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body, html {height: 100%; margin: 0; padding: 0;}
    .container {display: flex; flex-direction: column; height: 100%;}
    .panel {flex: 1; width: 100%; border: 1px solid #ccc;}
  </style>
  <title>Koogu | Model performance</title>
</head>
<body>
  <div class="container">
    <div class="panel" id="overall"></div>
    <div class="panel" id="perClass"></div>
  </div>

  <script>
    const layout = {
      hovermode: "closest",
      margin: {b: 40, l: 60, r: 10, t: 30},
      template: {data: {scatter: [{type: "scatter", mode: "lines+markers",
        text: [""")

        outf.write(','.join(f'{th:f}' for th in thresholds))

        outf.write("""],
        showlegend: true, hovertemplate: "<b>Threshold: </b>%{text}<br>" +
          "<b>Precision: </b>%{y:.4f}<br><b>Recall: </b>%{x:.4f}"}]}},
      xaxis: {range: [0.0, 1.0], title: {text: "Recall"}},
      yaxis: {range: [0.0, 1.0], title: {text: "Precision"}},
    };
    const config = {
      displayModeBar: "hover", displaylogo: false,
      responsive: true
    };
    Plotly.newPlot("overall", [
        {type: "scatter", showlegend: false, name: "All classes", """)

        outf.write(
            'x: [' +
            (','.join(json.dumps(float(v)) for v in overall_pr['recall'])) +
            '], ')
        outf.write(
            'y: [' +
            (','.join(json.dumps(float(v)) for v in overall_pr['precision'])) +
            ']')

        outf.write("""}
      ],
      {...layout, title: "Overall Performance"},
      {...config, toImageButtonOptions: {filename: """)
        outf.write(f'"{outfname}_Overall_PR"')
        outf.write("""}});
    Plotly.newPlot("perClass", [
        """)

        for cl, pr in per_class_pr.items():
            outf.write(f'{{type: "scatter", name: "{cl}", ')
            outf.write(
                'x: [' +
                (','.join(json.dumps(float(v)) for v in pr['recall'])) +
                '], '
            )
            outf.write(
                'y: [' +
                (','.join(json.dumps(float(v)) for v in pr['precision'])) +
                ']'
            )
            outf.write('},\n')

        outf.write("""
      ],
      {...layout, title: "Per-class Performance"},
      {...config, toImageButtonOptions: {filename: """)
        outf.write(f'"{outfname}_PerClass_PRs"')
        outf.write("""}});
  </script>
</body>
</html>
        """)
