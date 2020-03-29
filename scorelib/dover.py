"""Functions for combining multiple RTTMs into one using a DOVER variant."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict, namedtuple

from intervaltree import Interval, IntervalTree

import numpy as np
from scipy.linalg import block_diag

from . import metrics
from .six import iteritems, itervalues
from .utils import groupby

__all__ = ['combine_turns_list']

def combine_turns_list(turns_list, file_ids):
    # Build contingency matrices.
    file_to_turns_list = {}
    for turns in turns_list:
        for fid, g in groupby(turns, lambda x: x.file_id):
            if fid in file_to_turns_list:
                file_to_turns_list[fid].append(g)
            else:
                file_to_turns_list[fid] = [g]
    
    file_to_mapped_turns_list = get_mapped_turns_list(file_to_turns_list)
    
def get_mapped_turns_list(file_to_turns_list):
    file_to_mapped_turns_list = {}
    for file_id in file_to_turns_list.keys():
        turns_list = file_to_turns_list[file_id]
        file_to_mapped_turns_list[file_id] = [turns_list[0]]
        ref_turns = turns_list[0]
        for sys_turn in turns_list[1:]:
            mapped_sys_turns = get_mapped_turn(ref_turns, sys_turns)
            file_to_mapped_turns_list[file_id].append(mapped_sys_turns)
    return  file_to_mapped_sys_turns

def get_mapped_turns(ref_turns, sys_turns):
    cost = []
    ref_map = {}
    sys_map = {}
    for i, ref_spk_id, ref_spk_turns in enumerate(groupby(ref_turns, lambda x: x.speaker_id)):
        ref_map[i] = ref_spk_id
        cost.append([])
        for j, sys_spk_id, sys_spk_turns in enumerate(groupby(sys_turns, lambda x: x.speaker_id)):
            sys_map[j] = sys_spk_id
            cost[i].append(-1*compute_spk_overlap(ref_spk_turns, sys_spk_turns))
    cost = np.array(cost)
    print(cost)

def compute_spk_overlap(ref_spk_turns, sys_spk_turns):
    ref_tree = IntervalTree.from_tuples([(turn.onset,turn.offset) for turn in ref_spk_turns])
    sys_tree = IntervalTree.from_tuples([(turn.onset,turn.offset) for turn in sys_spk_turns])
    overlap_tree = ref_tree & sys_tree
    overlap_duration = sum([iv.end-iv.begin for iv in overlap_tree])
    return overlap_duration


