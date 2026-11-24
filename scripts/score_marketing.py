"""Score marketing start-time predictions against integer-second QA labels.

Task metric (per the "decision model 时序任务测试" wiki): the predicted
marketing time is the left edge of the highest-scoring temporal cell,
``k = argmax(logits); start_time = score_bounds[k][0]``. Accuracy is measured
at the 3-second tolerance (primary) and the stricter 0.5-second tolerance,
together with mean absolute error over all videos.
"""
import argparse
import csv
import json
from pathlib import Path

TOLERANCES = (0.5, 3.0)


def read_jsonl(path):
    """Load newline-delimited JSON records.

    Args:
        path (Path): JSONL file (manifest or model predictions).

    Returns:
        list[dict]: one object per nonempty line, in file order.

    Raises:
        ValueError: if a line is not a JSON object.
    """
    records = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{number}: invalid JSON: {error}') from error
        if not isinstance(record, dict):
            raise ValueError(f'{path}:{number}: each line must be a JSON object')
        records.append(record)
    return records


def predicted_start(prediction):
    """Apply the wiki single-point rule to one grounding prediction.

    Args:
        prediction (dict): prediction row containing ``score_bounds`` and
            either ``logits`` (preferred) or ``probabilities`` per cell.

    Returns:
        float: left edge (seconds) of the highest-scoring temporal cell.

    Raises:
        KeyError: when ``score_bounds`` is absent (Route B output required).
        ValueError: on empty cells or length mismatch between scores and bounds.
    """
    bounds = prediction.get('score_bounds')
    if bounds is None:
        raise KeyError(f"{prediction.get('id')}: score_bounds missing; rerun with Route B output")
    scores = prediction.get('logits')
    if scores is None:
        scores = prediction['probabilities']
    if not scores or len(scores) != len(bounds):
        raise ValueError(f"{prediction.get('id')}: scores and score_bounds length mismatch")
    index = max(range(len(scores)), key=scores.__getitem__)
    return float(bounds[index][0])


def score(manifest_rows, predictions):
    """Compute MAE and tolerance accuracies over the full marketing set.

    Args:
        manifest_rows (list[dict]): manifest rows carrying
            ``marketing_start_time`` integer-second GT.
        predictions (list[dict]): model prediction rows indexed by ``id``.

    Returns:
        dict: ``n``, ``mae``, ``acc@0.5``, ``acc@3`` and per-video details.

    Raises:
        ValueError: on id-set mismatch, missing GT, or malformed predictions.
    """
    labels = {row['id']: row for row in manifest_rows}
    by_id = {row['id']: row for row in predictions}
    if set(labels) != set(by_id) or len(by_id) != len(predictions):
        raise ValueError('Prediction and manifest id sets differ or prediction ids duplicate')
    details, errors = [], []
    for row_id, label_row in labels.items():
        gt = label_row.get('marketing_start_time')
        if type(gt) is not int:
            raise ValueError(f'{row_id}: marketing_start_time must be an integer GT')
        pred = predicted_start(by_id[row_id])
        error = abs(pred - gt)
        errors.append(error)
        details.append(dict(id=row_id, gt=gt, pred=pred, abs_error=round(error, 6)))
    report = dict(n=len(errors), mae=round(sum(errors) / len(errors), 6))
    for tolerance in TOLERANCES:
        report[f'acc@{tolerance}'] = round(
            sum(error <= tolerance for error in errors) / len(errors), 6)
    report['details'] = details
    return report


def cross_check_official(report, official_csv):
    """Compare extracted start times with the wiki's published videojev column.

    Args:
        report (dict): output of :func:`score`, updated in place with the
            exact-match count and the reference MAE.
        official_csv (Path): three-group CSV with ``vid``, ``GT`` and
            ``videojev`` columns.

    Returns:
        None. Adds an ``official_cross_check`` block to ``report``.

    Raises:
        ValueError: when an official vid is absent from the predictions.
    """
    with official_csv.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    ours = {item['id']: item for item in report['details']}
    exact, reference_errors, mismatches = 0, [], []
    for row in rows:
        vid = row['vid'].strip()
        if vid not in ours:
            raise ValueError(f'{vid}: present in official CSV but missing from predictions')
        official_pred = float(row['videojev'])
        reference_errors.append(abs(official_pred - float(row['GT'])))
        if abs(ours[vid]['pred'] - official_pred) <= 1e-9:
            exact += 1
        else:
            mismatches.append(dict(id=vid, ours=ours[vid]['pred'], official=official_pred))
    report['official_cross_check'] = dict(
        csv=str(official_csv), exact_start_time_matches=f'{exact}/{len(rows)}',
        reference_mae=round(sum(reference_errors) / len(reference_errors), 6),
        mismatches=mismatches)


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): explicit arguments; ``None`` reads ``sys.argv``.

    Returns:
        argparse.Namespace: validated CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='Model predictions JSONL (rank0.jsonl/predictions.jsonl)')
    parser.add_argument('--manifest', required=True, type=Path,
                        help='Marketing manifest JSONL with marketing_start_time GT')
    parser.add_argument('--official', type=Path,
                        help='Optional three-group CSV for videojev column cross-check')
    parser.add_argument('--out', type=Path, help='Optional path to write the full JSON report')
    return parser.parse_args(argv)


def main(argv=None):
    """Run scoring and print the summary.

    Args:
        argv (list[str] | None): explicit CLI arguments for testing.

    Returns:
        None. Prints the JSON report; writes it to ``--out`` when provided.
    """
    args = parse_args(argv)
    report = score(read_jsonl(args.manifest), read_jsonl(args.predictions))
    if args.official is not None:
        cross_check_official(report, args.official)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {key: value for key, value in report.items() if key != 'details'}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
