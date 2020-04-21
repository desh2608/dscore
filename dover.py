#!/usr/bin/env python3
"""This script implements a new version of the recently
proposed DOVER algorithm for combining diarization outputs,
with the difference that it is designed to work on
outputs containing overlapping segments, such as those
generated using EEND. (The original DOVER ignores overlaps)

Author: Desh Raj (Johns Hopkins University)
"""

from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys

from itertools import groupby

from scorelib import __version__ as VERSION
from scorelib.argparse import ArgumentParser
from scorelib.rttm import load_rttm
from scorelib.turn import merge_turns, trim_turns
from scorelib.score import score
from scorelib.six import iterkeys
from scorelib.uem import gen_uem, load_uem
from scorelib.utils import error, info, warn, xor
from scorelib.dover import combine_turns_list

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def load_rttms(rttm_list):
    """Loads speaker turns from input RTTMs in a list of turns."""
    turns = []
    file_ids = []
    for rttm_fn in rttm_list:
        if not os.path.exists(rttm_fn):
            error('Unable to open RTTM file: %s' % rttm_fn)
            sys.exit(1)
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns.append(turns_)
            file_ids += list(file_ids_)
        except IOError as e:
            error('Invalid RTTM file: %s. %s' % (rttm_fn, e))
            sys.exit(1)
    return turns, set(file_ids)

def check_for_empty_files(ref_turns, uem):
    """Warn on files in UEM without reference or speaker turns."""
    ref_file_ids = {turn.file_id for turn in ref_turns}
    for file_id in sorted(iterkeys(uem)):
        if file_id not in ref_file_ids:
            warn('File {} missing in input RTTM.'.format(file_id))
    # TODO: Clarify below warnings; this indicates that there are no
    #       ELIGIBLE reference/system turns.
    if not ref_turns:
        warn('No reference speaker turns found within UEM scoring regions.')

def main():
    """Main."""
    # Parse command line arguments.
    parser = ArgumentParser(
        description='Apply new DOVER on diarization outputs.', add_help=True,
        usage='%(prog)s [options]')
    parser.add_argument(
        '-i', nargs='+', dest='fin', help='Input RTTM files')
    parser.add_argument(
        '-o', nargs='?', default='rttm_out', dest='fout',
        help='output RTTM file')
    parser.add_argument(
        '-t,--threshold', type=float, default=0.5, dest='threshold',
        help='Speaker segment is retained if at least this fraction '
        'of input RTTMs agree on it')
    parser.add_argument(
        '-u,--uem', nargs=None, metavar='STR', dest='uemf',
        help='un-partitioned evaluation map file (default: %(default)s)')
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Load speaker/reference speaker turns and UEM. If no UEM specified,
    # determine it automatically.
    info('Loading speaker turns from input RTTMs...', file=sys.stderr)
    turns_list, file_ids = load_rttms(args.fin)
    if args.uemf is not None:
        info('Loading universal evaluation map...', file=sys.stderr)
        uem = load_uem(args.uemf)
    else:
        warn('No universal evaluation map specified. Approximating from '
             'input RTTM turn extents...')
        all_turns = [turn for sublist in turns_list for turn in sublist]
        uem = gen_uem(all_turns, all_turns)

    # Trim turns to UEM scoring regions and merge any that overlap.
    info('Trimming reference speaker turns to UEM scoring regions...',
         file=sys.stderr)
    turns_list = [trim_turns(turns, uem) for turns in turns_list]
    info('Checking for overlapping reference speaker turns...',
         file=sys.stderr)
    turns_list = [merge_turns(turns) for turns in turns_list]

    for turns in turns_list:
        check_for_empty_files(turns, uem)
    file_to_out_turns = combine_turns_list(turns_list, file_ids, args.threshold)

    # Write output RTTM file
    fmt = "SPEAKER {:s} 1 {:7.3f} {:7.3f} <NA> <NA> {:s} <NA>\n"
    with open(args.fout, 'w') as fh:
        for file_id in file_to_out_turns.keys():
            turns = merge_turns(file_to_out_turns[file_id])
            for turn in turns:
                fh.write(fmt.format(file_id, turn.onset, turn.dur, turn.speaker_id))
    return


if __name__ == '__main__':
    main()
