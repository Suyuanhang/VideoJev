"""Convert TimeLens-100K / TimeLens2-93K annotations into VideoJev JSONL manifests.

Each official annotation becomes one ``grounding`` row consumed by
``timejev.user_data.load_manifest``. Rows belonging to one video stay consecutive
and share one ``video_key``, matching the query-grouping contract of the
evaluation pipeline. No label text is rewritten beyond the official
``parse_query`` normalization.
"""
import argparse
import json
import math
import os
import random
import re
from pathlib import Path

PARSER_SEED = 20260924
LONG_VIDEO_SECONDS = 512.0


def parse_query(query):
    """Normalize a sentence exactly like the official TimeLens ``parse_query``.

    Args:
        query (str): raw annotation sentence.

    Returns:
        str: whitespace-collapsed text with surrounding whitespace and
        trailing periods removed.
    """
    return re.sub(r'\s+', ' ', query).strip().strip('.').strip()


def read_jsonl(path):
    """Load newline-delimited JSON records.

    Args:
        path (Path): official annotation ``.jsonl`` file.

    Returns:
        list[dict]: one parsed object per nonempty line, in file order.

    Raises:
        ValueError: if a line is not a JSON object.
    """
    records = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
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


def select_videos(keys, max_videos, seed, strata=None):
    """Deterministically choose a capped, optionally stratified video subset.

    Args:
        keys (list[str]): video identifiers in their official order.
        max_videos (int): positive cap; non-positive values keep every video.
        seed (int): base RNG seed making the selection reproducible.
        strata (dict[str, str] | None): identifier to stratum mapping; when
            given, strata are sampled round-robin after independent shuffles.

    Returns:
        list[str]: selected identifiers, returned in the original ``keys`` order.
    """
    if not max_videos or max_videos >= len(keys):
        return list(keys)
    if strata is None:
        pool = list(keys)
        random.Random(seed).shuffle(pool)
        chosen = pool[:max_videos]
    else:
        names = sorted(set(strata.values()))
        pools = {name: [] for name in names}
        for key in keys:
            pools[strata[key]].append(key)
        for index, name in enumerate(names):
            random.Random(seed + index).shuffle(pools[name])
        chosen, cursor = [], 0
        while len(chosen) < max_videos and any(pools.values()):
            name = names[cursor % len(names)]
            if pools[name]:
                chosen.append(pools[name].pop(0))
            cursor += 1
    order = {key: index for index, key in enumerate(keys)}
    return sorted(chosen, key=order.get)


def clean_spans(pairs):
    """Validate timestamp pairs and drop annotations without a positive length.

    Args:
        pairs (list[list[float]]): raw ``[start, end]`` pairs from the annotation.

    Returns:
        tuple[list[list[float]], int]: the kept finite spans in their original
        order plus the number of zero/negative-length pairs dropped.

    Raises:
        ValueError: if an endpoint is non-numeric or non-finite.
    """
    kept, dropped = [], 0
    for pair in pairs:
        start, end = float(pair[0]), float(pair[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f'non-finite grounding span: {pair}')
        if start < end:
            kept.append([start, end])
        else:
            dropped += 1
    return kept, dropped


def probe_duration(path):
    """Read container-level video duration without decoding frames.

    Args:
        path (Path): media file readable by PyAV.

    Returns:
        float: stream duration in seconds, falling back to container duration
        when the stream does not report one. Mirrors
        ``videojev_hf.processing_videojev.prepare_video``.

    Raises:
        av.FFmpegError: if the container cannot be opened or has no video stream.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration) / 1e6


def make_row(row_id, video_key, media, out_dir, duration, query, spans):
    """Build one manifest row with a media path relative to the manifest.

    Args:
        row_id (str): globally unique question identifier.
        video_key (str): identifier shared by every question about one video.
        media (Path): absolute media path on this host.
        out_dir (Path): directory that will contain the manifest.
        duration (float): media duration in seconds.
        query (str): normalized question text.
        spans (list[list[float]]): ground-truth ``[start, end]`` intervals.

    Returns:
        dict: a row accepted by ``timejev.user_data.load_manifest``.
    """
    return dict(id=row_id, video_key=video_key,
                video_path=os.path.relpath(media, out_dir),
                duration=duration, task='grounding', query=query, spans=spans)


def convert_timelens100k(records, video_root, out_dir, max_videos, seed, sources, allowed_ids=None):
    """Convert TimeLens-100K records (one object per video, ``events`` list).

    Args:
        records (list[dict]): parsed ``timelens-100k.jsonl`` objects.
        video_root (Path): directory containing the extracted ``<source>/<key>.mp4`` tree.
        out_dir (Path): target manifest directory for relative media paths.
        max_videos (int): optional per-run video cap.
        seed (int): subset RNG seed.
        sources (set[str]): empty set keeps all sources; otherwise filters first.
        allowed_ids (set[str] | None): optional allowlist matched against each
            record's ``video_path`` or its filename stem; restricts the manifest
            to media actually present in a partial download.

    Returns:
        tuple[list[dict], dict]: manifest rows and a conversion statistics dict.

    Raises:
        ValueError: on duplicated video paths, invalid records or missing media.
    """
    if sources:
        records = [r for r in records if r.get('source') in sources]
    if allowed_ids is not None:
        records = [r for r in records
                   if r['video_path'] in allowed_ids or Path(r['video_path']).stem in allowed_ids]
    keys = [r['video_path'] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('TimeLens-100K annotations contain duplicated video_path values')
    strata = {r['video_path']: r['source'] for r in records}
    selected = set(select_videos(keys, max_videos, seed, strata))
    stats = dict(videos_total=len(keys), videos_selected=len(selected), rows_written=0,
                 spans_dropped_zero_length=0, queries_dropped=0, videos_without_rows=0,
                 long_videos_gt512s=0, missing_media=[])
    rows = []
    for record in records:
        if record['video_path'] not in selected:
            continue
        media = (video_root / record['video_path']).resolve()
        if not media.is_file():
            stats['missing_media'].append(str(media))
            continue
        duration = float(record['duration'])
        if duration > LONG_VIDEO_SECONDS:
            stats['long_videos_gt512s'] += 1
        video_key = f"{record['source']}/{Path(record['video_path']).stem}"
        before = len(rows)
        for index, event in enumerate(record['events']):
            query = parse_query(event['query'])
            spans, dropped = clean_spans(event['span'])
            stats['spans_dropped_zero_length'] += dropped
            if not spans or not query or '<|' in query:
                stats['queries_dropped'] += 1
                continue
            rows.append(make_row(f'{video_key}#{index:03d}', video_key, media,
                                 out_dir, duration, query, spans))
        if len(rows) == before:
            stats['videos_without_rows'] += 1
    stats['rows_written'] = len(rows)
    return rows, stats


def convert_timelens2(records, video_root, out_dir, max_videos, seed, video_template,
                      allowed_ids=None):
    """Convert TimeLens2-93K raw annotations (one object per video).

    Args:
        records (list[dict]): parsed ``TimeLens2-93K_raw_annotations.jsonl`` objects.
        video_root (Path): directory holding the media extracted from the 18 tar shards.
        out_dir (Path): target manifest directory for relative media paths.
        max_videos (int): optional per-run video cap.
        seed (int): subset RNG seed.
        video_template (str): filename template containing one ``{video_id}`` field.
        allowed_ids (set[str] | None): optional ``video_id`` allowlist restricting
            the manifest to media extracted from a partial tar download.

    Returns:
        tuple[list[dict], dict]: manifest rows and a conversion statistics dict.

    Raises:
        ValueError: on duplicated IDs, invalid records, missing media or probe failure.
    """
    if allowed_ids is not None:
        records = [r for r in records if r['video_id'] in allowed_ids]
    keys = [r['video_id'] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('TimeLens2 annotations contain duplicated video_id values')
    selected = set(select_videos(keys, max_videos, seed))
    stats = dict(videos_total=len(keys), videos_selected=len(selected), rows_written=0,
                 spans_dropped_zero_length=0, queries_dropped=0, videos_without_rows=0,
                 long_videos_gt512s=0, missing_media=[])
    rows = []
    for record in records:
        video_id = record['video_id']
        if video_id not in selected:
            continue
        media = (video_root / video_template.format(video_id=video_id)).resolve()
        if not media.is_file():
            stats['missing_media'].append(str(media))
            continue
        duration = probe_duration(media)
        if duration > LONG_VIDEO_SECONDS:
            stats['long_videos_gt512s'] += 1
        before = len(rows)
        for index, annotation in enumerate(record['annotations']):
            query = parse_query(annotation['text'])
            spans, dropped = clean_spans(annotation['timestamps'])
            stats['spans_dropped_zero_length'] += dropped
            if not spans or not query or '<|' in query:
                stats['queries_dropped'] += 1
                continue
            rows.append(make_row(f'{video_id}#{index:03d}', video_id, media,
                                 out_dir, duration, query, spans))
        if len(rows) == before:
            stats['videos_without_rows'] += 1
    stats['rows_written'] = len(rows)
    return rows, stats


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): explicit arguments; ``None`` reads ``sys.argv``.

    Returns:
        argparse.Namespace: validated CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('timelens100k', 'timelens2'))
    parser.add_argument('--annotations', required=True, type=Path,
                        help='Official annotation jsonl (timelens-100k.jsonl or TimeLens2-93K_raw_annotations.jsonl)')
    parser.add_argument('--video-root', required=True, type=Path,
                        help='Directory containing the extracted video files')
    parser.add_argument('--out', required=True, type=Path, help='Output manifest jsonl path')
    parser.add_argument('--max-videos', type=int, default=0,
                        help='Cap the number of videos (deterministic subset); 0 keeps all')
    parser.add_argument('--seed', type=int, default=PARSER_SEED)
    parser.add_argument('--sources', default='',
                        help='TimeLens-100K only: comma-separated source allowlist, e.g. didemo,queryd')
    parser.add_argument('--video-template', default='{video_id}.mp4',
                        help='TimeLens2 only: media filename template relative to --video-root')
    parser.add_argument('--video-ids-file', type=Path,
                        help='Optional text file with one available video id/stem per line; '
                             'restricts output to media present in a partial shard download')
    return parser.parse_args(argv)


def read_id_file(path):
    """Load an optional one-id-per-line allowlist.

    Args:
        path (Path | None): text file with video ids or filename stems.

    Returns:
        set[str] | None: trimmed nonempty lines, or None when no path is given.
    """
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def main(argv=None):
    """Run the conversion and write the manifest plus a stdout summary.

    Args:
        argv (list[str] | None): explicit CLI arguments for testing.

    Returns:
        None. Writes JSONL to ``--out`` and prints statistics as JSON.

    Raises:
        SystemExit: via argparse; ValueError propagates on any data/media error,
        in which case no manifest is written.
    """
    args = parse_args(argv)
    records = read_jsonl(args.annotations)
    out_dir = args.out.resolve().parent
    sources = {item.strip() for item in args.sources.split(',') if item.strip()}
    allowed_ids = read_id_file(args.video_ids_file)
    if args.dataset == 'timelens100k':
        rows, stats = convert_timelens100k(records, args.video_root.resolve(), out_dir,
                                           args.max_videos, args.seed, sources, allowed_ids)
    else:
        if sources:
            raise ValueError('--sources is only valid for TimeLens-100K')
        rows, stats = convert_timelens2(records, args.video_root.resolve(), out_dir,
                                        args.max_videos, args.seed, args.video_template,
                                        allowed_ids)
    if stats['missing_media']:
        missing = stats['missing_media']
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} selected videos are missing under {args.video_root}; first: {preview}')
    if not rows:
        raise ValueError('No valid rows produced; refusing to write an empty manifest')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                for row in rows), encoding='utf-8')
    stats.update(dataset=args.dataset, manifest=str(args.out.resolve()))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
