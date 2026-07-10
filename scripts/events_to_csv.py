#!/usr/bin/env python3
"""Convert TensorFlow Events to CSV files for a training run."""

import os
import csv
from tensorboard.backend.event_processing import event_accumulator

def events_to_csv(tfevents_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    ea = event_accumulator.EventAccumulator(tfevents_path)
    ea.Reload()

    all_tags = [t for t in ea.Tags()['scalars']]

    # Group tags by prefix
    def prefix(tag):
        return tag.split('/')[0]

    prefixes = sorted(set(prefix(t) for t in all_tags))

    # Build per-tag time-series
    tag_series = {}
    for tag in all_tags:
        events = ea.Scalars(tag)
        tag_series[tag] = [(e.step, e.wall_time, e.value) for e in events]

    # Determine global step range from any non-empty series
    global_steps = None
    for tag, series in tag_series.items():
        if series:
            steps = [x[0] for x in series]
            if global_steps is None:
                global_steps = steps
            else:
                global_steps = sorted(set(global_steps) | set(steps))

    print(f"Global step range: {min(global_steps)} → {max(global_steps)}, {len(global_steps)} unique steps")

    # Build lookup: (tag -> {step: value})
    lookup = {}
    for tag, series in tag_series.items():
        lookup[tag] = {s: v for s, _, v in series}

    # --- Write split CSV files ---

    def write_csv(filename, tags):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'wall_time'] + tags)
            for step in global_steps:
                row = [step]
                wall_time = None
                for tag in tags:
                    if tag in lookup and step in lookup[tag]:
                        if wall_time is None:
                            for s, wt, v in tag_series[tag]:
                                if s == step:
                                    wall_time = wt
                                    break
                        row.append(lookup[tag][step])
                    else:
                        row.append('')
                if wall_time is not None or any(row[2:]):
                    writer.writerow(row)
        print(f"  Wrote {filename} ({len(tags)} columns, {len(global_steps)} rows)")

    # Group 1: loss
    loss_tags = sorted([t for t in all_tags if prefix(t) == 'Loss'])
    write_csv('loss.csv', loss_tags)

    # Group 2: episode reward components
    ep_tags = sorted([t for t in all_tags if prefix(t) == 'Episode_Reward'])
    write_csv('episode_reward.csv', ep_tags)

    # Group 3: episode termination
    term_tags = sorted([t for t in all_tags if prefix(t) == 'Episode_Termination'])
    write_csv('episode_termination.csv', term_tags)

    # Group 4: curriculum
    curr_tags = sorted([t for t in all_tags if prefix(t) == 'Curriculum'])
    write_csv('curriculum.csv', curr_tags)

    # Group 5: metrics
    met_tags = sorted([t for t in all_tags if prefix(t) == 'Metrics'])
    write_csv('metrics.csv', met_tags)

    # Group 6: perf
    perf_tags = sorted([t for t in all_tags if prefix(t) == 'Perf'])
    write_csv('perf.csv', perf_tags)

    # Group 7: train summary
    train_tags = sorted([t for t in all_tags if prefix(t) == 'Train'])
    write_csv('train_summary.csv', train_tags)

    # Group 8: policy
    policy_tags = sorted([t for t in all_tags if prefix(t) == 'Policy'])
    write_csv('policy.csv', policy_tags)

    # Group 9: all scalars (one big CSV)
    write_csv('all_scalars.csv', all_tags)

    print(f"\nDone. All CSVs written to {output_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 events_to_csv.py <path_to_events> <output_dir>")
        sys.exit(1)
    events_to_csv(sys.argv[1], sys.argv[2])
