"""Functions for combining multiple RTTMs into one using a DOVER variant."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict, namedtuple

from intervaltree import Interval, IntervalTree

import numpy as np
import itertools, sys
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment

from . import metrics
from .six import iteritems, itervalues
from .utils import groupby
from .turn import Turn

__all__ = ['combine_turns_list']

def combine_turns_list(turns_list, file_ids, threshold):
    # Build contingency matrices.
    file_to_turns_list = {}
    for turns in turns_list:
        for fid, g in groupby(turns, lambda x: x.file_id):
            if fid in file_to_turns_list:
                file_to_turns_list[fid].append(list(g))
            else:
                file_to_turns_list[fid] = [list(g)]

    # First we map speaker labels in all RTTMs using the Hungarian algorithm
    print ("Mapping speaker labels..")
    file_to_mapped_turns_list = get_mapped_turns_list(file_to_turns_list)

    # Now we apply the speaker-wise DOVER to combine the turns list for
    # each file (here, file means a session id or a recording id)
    print ("Applying DOVER-Lap")
    file_to_combined_turns = get_combined_turns(file_to_mapped_turns_list, threshold)

    return file_to_combined_turns
   
def get_combined_turns(file_to_turns_list, threshold):
    """
    This function takes turns list for all input RTTMs and performs
    the new DOVER-Lap for getting a combined output turns list.
    """
    file_to_combined_turns = {}
    for file_id in file_to_turns_list.keys():
        turns_list = file_to_turns_list[file_id]
        combined_turns = doverlap(turns_list, file_id, threshold)
        file_to_combined_turns[file_id] = combined_turns
    return file_to_combined_turns

def doverlap(turns_list, file_id, threshold):
    """
    This method implements the actual DOVER-Lap algorithm
    """
    num_files = len(turns_list)
    all_combined_turns = []
    spk_to_turns_list = {}
    for turns in turns_list:
        groups = {key:list(group) for key, group in itertools.groupby(turns, lambda x: x.speaker_id)}
        for key in groups.keys():
            group = groups[key] 
            if key in spk_to_turns_list:
                spk_to_turns_list[key].append(group)
            else:
                spk_to_turns_list[key] = [group]
    
    for spk_id in spk_to_turns_list.keys():
        spk_turns = spk_to_turns_list[spk_id]
        combined_turns = combine_speaker_turns(spk_turns, spk_id, file_id, threshold)
        all_combined_turns += combined_turns
    
    return all_combined_turns

def combine_speaker_turns(turns_list, spk_id, file_id, threshold = 0.5):
    """
    Given segments from different RTTMs for a single speaker, return
    their intersection, such that at least _threshold_ fraction of
    RTTMs must agree on a segment.
    """
    start_time = sys.maxsize
    end_time = 0
    num_files = len(turns_list)
    trees = []
    combined_turns = []
    time_marks = set()
    for turns in turns_list:
        cur_start = list(sorted(turns, key=lambda x: x.onset))[0].onset
        cur_end = list(sorted(turns, key=lambda x: x.onset))[-1].offset
        start_time = min(start_time, cur_start)
        end_time = max(end_time, cur_end)
        trees.append(IntervalTree(Interval(turn.onset,turn.offset) for turn in turns))
        for turn in turns:
            time_marks.add(turn.onset)
            time_marks.add(turn.offset)

    sorted_time_marks = list(sorted(time_marks))
    for i in range(len(sorted_time_marks)-1):
        cur_start = sorted_time_marks[i]
        cur_end = sorted_time_marks[i+1]
        num_present = 0
        for tree in trees:
            if tree.overlap(cur_start, cur_end):
                num_present += 1
        if (num_present >= threshold*num_files):
            combined_turns.append(Turn(cur_start, cur_end, speaker_id=spk_id, file_id=file_id))
    return combined_turns


def get_mapped_turns_list(file_to_turns_list):
    """
    This function takes turns list from all RTTMs and performs
    Hungarian algorithm on the speaker labels to get the best possible
    mapping which maximizes overlap duration.
    """
    file_to_mapped_turns_list = {}
    for file_id in file_to_turns_list.keys():
        turns_list = file_to_turns_list[file_id]
        min_cost = sys.maxsize
        best_ref = 0
        for i, ref_turns in enumerate(turns_list):
            print ("Mapping using {} as reference".format(i))
            cur_cost = 0
            mapped_turns_list = [ref_turns]
            for sys_turns in turns_list[1:]:
                mapped_sys_turns, cost = get_mapped_turns(ref_turns, sys_turns)
                mapped_turns_list.append(mapped_sys_turns)
                cur_cost += cost
            print ("Cost for file {} with ref {}: {}".format(file_id, i, cur_cost))
            if cur_cost < min_cost:
                min_cost = cur_cost
                file_to_mapped_turns_list[file_id] = mapped_turns_list
                best_ref = i
        print ("{} is the best reference for file {}".format(best_ref, file_id))
    return file_to_mapped_turns_list

def get_mapped_turns(ref_turns, sys_turns):
    cost = []
    ref_groups = {key: list(group) for key, group in itertools.groupby(ref_turns, lambda x: x.speaker_id)};
    sys_groups = {key: list(group) for key, group in itertools.groupby(sys_turns, lambda x: x.speaker_id)};
    for ref_spk_id in ref_groups.keys():
        cur_row = []
        ref_spk_turns = ref_groups[ref_spk_id]
        for sys_spk_id in sys_groups.keys():
            sys_spk_turns = sys_groups[sys_spk_id]
            total_overlap = compute_spk_overlap(ref_spk_turns, sys_spk_turns)
            cur_row.append(-1*total_overlap)
        cost.append(cur_row)
    cost = np.array(cost)
    ref_spk_ids_map = {i:j for i,j in enumerate(ref_groups.keys())}
    sys_spk_ids_revmap = {j:i for i,j in enumerate(sys_groups.keys())}
    best_map, min_cost = weighted_bipartite_graph_match(cost)
    mapped_sys_turns = []
    for turn in sys_turns:
        old_spk_id = turn.speaker_id
        new_spk_id = ref_spk_ids_map[best_map[sys_spk_ids_revmap[old_spk_id]]]
        turn.speaker_id = new_spk_id
        mapped_sys_turns.append(turn)
    return mapped_sys_turns, min_cost

def weighted_bipartite_graph_match(cost):
    tmp, ref_to_sys = linear_sum_assignment(cost)
    best_map = {j:i for i,j in enumerate(ref_to_sys)}
    min_cost = cost[tmp, ref_to_sys].sum()
    return best_map, min_cost

def compute_spk_overlap(ref_spk_turns, sys_spk_turns):
    ref_tree = IntervalTree.from_tuples([(turn.onset,turn.offset) for turn in ref_spk_turns])
    sys_tree = IntervalTree.from_tuples([(turn.onset,turn.offset) for turn in sys_spk_turns])
    ref_duration = sum([iv.end-iv.begin for iv in ref_tree])
    sys_duration = sum([iv.end-iv.begin for iv in sys_tree])
    combine_tree = ref_tree | sys_tree
    combine_tree.merge_overlaps(strict=False)
    combine_duration = sum([iv.end-iv.begin for iv in combine_tree])
    return (ref_duration + sys_duration - combine_duration)



    overlap_tree = find_intersection(ref_tree, sys_tree)
    overlap_duration = sum([iv.end-iv.begin for iv in overlap_tree])
    return overlap_duration


