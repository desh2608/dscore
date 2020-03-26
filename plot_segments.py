#!/usr/bin/env python3
"""Plot speaker segments from RTTM file
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

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

class RefRTTMAction(argparse.Action):
    """Custom action to ensure that reference files are specified from a
    script file or from the command line but not both.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if not xor(namespace.ref_rttm_fns, namespace.ref_rttm_scpf):
            parser.error('Exactly one of -r and -R must be set.')

def load_rttms(rttm_fns):
    """Load speaker turns from RTTM files.

    Parameters
    ----------
    rttm_fns : list of str
        Paths to RTTM files.

    Returns
    -------
    turns : list of Turn
        Speaker turns.

    file_ids : set
        File ids found in ``rttm_fns``.
    """
    turns = []
    file_ids = set()
    for rttm_fn in rttm_fns:
        if not os.path.exists(rttm_fn):
            error('Unable to open RTTM file: %s' % rttm_fn)
            sys.exit(1)
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns.extend(turns_)
            file_ids.update(file_ids_)
        except IOError as e:
            error('Invalid RTTM file: %s. %s' % (rttm_fn, e))
            sys.exit(1)
    return turns, file_ids

def check_for_empty_files(ref_turns, uem):
    """Warn on files in UEM without reference or speaker turns."""
    ref_file_ids = {turn.file_id for turn in ref_turns}
    for file_id in sorted(iterkeys(uem)):
        if file_id not in ref_file_ids:
            warn('File "%s" missing in reference RTTMs.' % file_id)
    # TODO: Clarify below warnings; this indicates that there are no
    #       ELIGIBLE reference/system turns.
    if not ref_turns:
        warn('No reference speaker turns found within UEM scoring regions.')

def plot_segments(ref_turns):
    colors = ['b','g','r','y']
    i = 0
    fig, ax = plt.subplots()
    for speaker_id, speaker_turns in groupby(ref_turns, lambda x: x.speaker_id):
        print ("Speaker {}".format(speaker_id))
        segments = []
        y = (i+2)
        for speaker_turn in speaker_turns:
            segments.append([(speaker_turn.onset,y),(
                speaker_turn.offset,y)])
        lc = mc.LineCollection(segments, linewidths=60, colors=colors[i])
        ax.add_collection(lc)
        i += 1
    ax.autoscale()
    plt.show()


def main():
    """Main."""
    # Parse command line arguments.
    parser = ArgumentParser(
        description='Plot speaker segments from an RTTM file.', add_help=True,
        usage='%(prog)s [options]')
    parser.add_argument(
        '-r', nargs='+', default=[], metavar='STR', dest='ref_rttm_fns',
        action=RefRTTMAction,
        help='reference RTTM files (default: %(default)s)')
    parser.add_argument(
        '-R', nargs=None, metavar='STR', dest='ref_rttm_scpf',
        action=RefRTTMAction,
        help='reference RTTM script file (default: %(default)s)')
    parser.add_argument(
        '-u,--uem', nargs=None, metavar='STR', dest='uemf',
        help='un-partitioned evaluation map file (default: %(default)s)')
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Check that at least one reference RTTM and at least one system RTTM
    # was specified.
    if args.ref_rttm_scpf is not None:
        args.ref_rttm_fns = load_script_file(args.ref_rttm_scpf)
    if not args.ref_rttm_fns:
        error('No reference RTTMs specified.')
        sys.exit(1)

    # Load speaker/reference speaker turns and UEM. If no UEM specified,
    # determine it automatically.
    info('Loading speaker turns from reference RTTMs...', file=sys.stderr)
    ref_turns, _ = load_rttms(args.ref_rttm_fns)
    if args.uemf is not None:
        info('Loading universal evaluation map...', file=sys.stderr)
        uem = load_uem(args.uemf)
    else:
        warn('No universal evaluation map specified. Approximating from '
             'reference turn extents...')
        uem = gen_uem(ref_turns, ref_turns)

    # Trim turns to UEM scoring regions and merge any that overlap.
    info('Trimming reference speaker turns to UEM scoring regions...',
         file=sys.stderr)
    ref_turns = trim_turns(ref_turns, uem)
    info('Checking for overlapping reference speaker turns...',
         file=sys.stderr)
    ref_turns = merge_turns(ref_turns)

    # Score.
    info('Scoring...', file=sys.stderr)
    check_for_empty_files(ref_turns, uem)
    plot_segments(ref_turns)


if __name__ == '__main__':
    main()