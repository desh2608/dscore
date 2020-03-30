#!/usr/bin/env python
"""
This takes 2 RTTMs (one from a diarization system and another
from a speech activity detection system) and applies the SAD
on the diarization output, i.e., it effectively removes the 
segments which the SAD system marked as non-speech.
"""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys

from scorelib import __version__ as VERSION
from scorelib.argparse import ArgumentParser
from scorelib.rttm import load_rttm
from scorelib.turn import Turn, merge_turns, trim_turns
from scorelib.score import score
from scorelib.six import iterkeys
from scorelib.uem import gen_uem, load_uem
from scorelib.utils import error, info, warn, xor, groupby

from intervaltree import Interval, IntervalTree
from collections import defaultdict, namedtuple

def check_for_empty_files(ref_turns, sys_turns, uem):
    """Warn on files in UEM without reference or speaker turns."""
    ref_file_ids = {turn.file_id for turn in ref_turns}
    sys_file_ids = {turn.file_id for turn in sys_turns}
    for file_id in sorted(iterkeys(uem)):
        if file_id not in ref_file_ids:
            warn('File "%s" missing in reference RTTMs.' % file_id)
        if file_id not in sys_file_ids:
            warn('File "%s" missing in system RTTMs.' % file_id)
    # TODO: Clarify below warnings; this indicates that there are no
    #       ELIGIBLE reference/system turns.
    if not ref_turns:
        warn('No reference speaker turns found within UEM scoring regions.')
    if not sys_turns:
        warn('No system speaker turns found within UEM scoring regions.')


def validate_nested_dict(nested_intervals):
    count = [0,0,0,0]
    for key, value in nested_intervals.items():
        if key.data != "overlap":
            if len(value) == 1:
                count[0] += 1
            else:
                count[3] += 1
        else:
            if len(value) == 1:
                count[1] += 1
            else:
                count[2] += 1
    print (count)
    return

def handle_overlaps(ref_tree, overlap_intervals):
    sorted_intervals = list(sorted(ref_tree.all_intervals, key=lambda x: x.begin))
    
    class Stats:
        num_added = 0
        num_expanded = 0

    def _handle_single_child_overlap(parent, child):
        current_index = sorted_intervals.index(parent)
        cur_spkr = parent.data
        try:
            prev_spkr = sorted_intervals[current_index - 1].data
            next_spkr = sorted_intervals[current_index + 1].data
            if (prev_spkr != cur_spkr):
                ref_tree.add(Interval(child.begin, child.end, prev_spkr))
                Stats.num_added += 1
            if (next_spkr != cur_spkr):
                ref_tree.add(Interval(child.begin, child.end, next_spkr))
                Stats.num_added += 1
        except IndexError:
            return
        return

    def _handle_multiple_child_overlap(parent, child_list):
        for child in child_list:
            _handle_single_child_overlap(parent, child)
        return

    def _handle_single_parent_overlap(parent, child):
        ref_tree.remove(Interval(child.begin, child.end, child.data))
        ref_tree.add(Interval(parent.begin, parent.end, child.data))
        Stats.num_expanded += 1
        return

    def _handle_multiple_parent_overlap(parent, child_list):
        for child in child_list:
            _handle_single_parent_overlap(parent, child)
        return

    for parent, children in overlap_intervals.items():
        if parent.data != "overlap": # Cases 1 and 4
            _handle_multiple_child_overlap(parent, list(children))
        else: # Cases 2 and 3
            _handle_multiple_parent_overlap(parent, list(children))
    
    return Stats.num_added, Stats.num_expanded
        
def get_overlap_regions(tree, interval):
    overlap_intervals = []
    sorted_tree = list(sorted(tree.all_intervals, key=lambda x: x.begin))
    for cur_int in sorted_tree:
        if cur_int.begin < interval.end and cur_int.end > interval.begin:
            overlap_begin = max(cur_int.begin, interval.begin)
            overlap_end = min(cur_int.end, interval.end)
            if (overlap_end > overlap_begin):
                overlap = Interval(overlap_begin, overlap_end, interval.data)
                overlap_intervals.append(overlap)
    return overlap_intervals

def remove_silence(ref_turns, sad_turns, file_ids):
    file_to_ref_turns = defaultdict(list,
        {fid : list(g) for fid, g in groupby(ref_turns, lambda x: x.file_id)})
    file_to_sad_turns = defaultdict(list,
        {fid : list(g) for fid, g in groupby(sad_turns, lambda x: x.file_id)})
    out_turns = []
    for fid in file_ids:
        turns = file_to_ref_turns[fid]
        speech = file_to_sad_turns[fid]
        ref_tree = IntervalTree(Interval(turn.onset,turn.offset,turn.speaker_id) for turn in turns)
        sad_tree = IntervalTree(Interval(turn.onset,turn.offset) for turn in speech)
        ref_tree_no_sil = IntervalTree()
        for interval in sad_tree:
            ref_envelop_intervals = ref_tree.envelop(interval.begin, interval.end)
            for ref_int in ref_envelop_intervals:
                assert (ref_int.end > ref_int.begin)
                ref_tree_no_sil.add(ref_int)
                ref_tree.remove(ref_int)
        for interval in ref_tree:
            overlap_intervals = get_overlap_regions(sad_tree, interval)
            for oi in overlap_intervals:
                assert (oi.end > oi.begin)
                ref_tree_no_sil.add(oi)
        for interval in list(ref_tree_no_sil):
            out_turns.append(Turn(interval.begin, offset=interval.end, speaker_id=interval.data, file_id=fid))
    return out_turns

def generate_rttm(turns, out_file):
    fmt = "SPEAKER {:s} 1 {:7.4f} {:7.4f} <NA> <NA> {:s} <NA>\n"
    with open(out_file, 'w') as fout:
        for turn in turns:
            fout.write(fmt.format(turn.file_id, turn.onset, turn.dur, turn.speaker_id))
    return

def main():
    """Main."""
    # Parse command line arguments.
    parser = ArgumentParser(
        description='Score diarization from RTTM files.', add_help=True,
        usage='%(prog)s [options]')
    parser.add_argument(
        '-r', metavar='STR', dest='ref_rttm',
        help='reference RTTM file')
    parser.add_argument(
        '-s', metavar='STR', dest='sad_rttm',
        help='SAD RTTM file')
    parser.add_argument(
        '-o', metavar='STR', default='output_rttm', dest='out_file',
        help='path to output file')
    parser.add_argument(
        '-u,--uem', nargs=None, metavar='STR', dest='uemf',
        help='un-partitioned evaluation map file (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if not args.ref_rttm:
        error('No reference RTTMs specified.')
        sys.exit(1)
    if not args.sad_rttm:
        error('No SAD RTTMs specified.')
        sys.exit(1)

    # Load speaker/reference speaker turns and UEM. If no UEM specified,
    # determine it automatically.
    info('Loading speaker turns from reference RTTM...', file=sys.stderr)
    ref_turns, _, file_ids = load_rttm(args.ref_rttm)
    info('Loading turns from overlap RTTM...', file=sys.stderr)
    sad_turns, _, _ = load_rttm(args.sad_rttm)
    if args.uemf is not None:
        info('Loading universal evaluation map...', file=sys.stderr)
        uem = load_uem(args.uemf)
    else:
        warn('No universal evaluation map specified. Approximating from '
             'reference and speaker turn extents...')
        uem = gen_uem(ref_turns, ref_turns)

    # Trim turns to UEM scoring regions and merge any that overlap.
    info('Trimming reference speaker turns to UEM scoring regions...',
         file=sys.stderr)
    ref_turns = trim_turns(ref_turns, uem)
    info('Checking for overlapping reference speaker turns...',
         file=sys.stderr)
    ref_turns = merge_turns(ref_turns)
    info('Trimming overlap turns to UEM scoring regions...',
         file=sys.stderr)
    sad_turns = trim_turns(sad_turns, uem)
    info('Checking for overlapping overlap turns...',
         file=sys.stderr)
    sad_turns = merge_turns(sad_turns)
    
    info('Creating output RTTM...', file=sys.stderr)
    check_for_empty_files(ref_turns, sad_turns, uem)
    ref_turns_without_sil = remove_silence(ref_turns, sad_turns, file_ids)
    dur_prev = sum([turn.dur for turn in ref_turns])
    dur_without_sil = sum([turn.dur for turn in ref_turns_without_sil])
    print ("Duration reduced from {} to {}".format(dur_prev, dur_without_sil))

    generate_rttm(ref_turns_without_sil, args.out_file)
    return


if __name__ == '__main__':
    main()
