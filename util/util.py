import sys
from collections import defaultdict
from uuid import uuid4
import numpy as np

def count_dict(depth=1) -> defaultdict:
    """
    Gets a defaultdict, that the default value is 0. 
    If the parameter depth is >1, this will be a defaultdict of such defaultdicts with depth depth.
    For example, count_dict(2)

    Parameters
    ----------
    depth : int

    Examples
    --------
    A count_dict of depth 2:
    >>> d = count_dict(2)
    >>> d[1][2] += 3
    >>> d[2][1]
    0
    >>> d[1][2]
    3

    A count _dict of depth 4:
    >>> d = count_dict(4)
    >>> d[1][2][3][4] += 3
    >>> d[1][2][3][1]
    0
    >>> d[1][2][3]
    {4:3, 1:0}
    """
    def _count_dict_inner(depth):
        if depth==1:
            return lambda : defaultdict(lambda : 0)
        return lambda : defaultdict(_count_dict_inner(depth-1))
    return _count_dict_inner(depth)()

get_unique_id = lambda : np.random.choice(sys.maxsize)

def max_fractional_difference(ext_T1, ext_T2):
    """
    Get the maximal change in transition probabilities between ext_T1 and ext_T2, where ext_T1/2
    are tuples of (ids, transition_probabilities), as returned by 
    HierarchicalMSMVertex.get_external_T.

    Parameters
    ----------
    ext_T1 : tuple
        the previous value of external_T
    ext_T2 : tuple
        the current value of external_T

    Returns
    -------
    the maximal difference in transition probabilities, as a fractional difference, that is:
    max[abs(1-(p1/p2)) for p1 in ext_T1 and p2 in ext_T2 with matching ids]
    """
    ids1, T1 = ext_T1
    ids2, T2 = ext_T2
    assert T1.shape == T2.shape, "external_T has changed shape since last update"
    sort1 = np.argsort(ids1) # sort by ids
    sort2 = np.argsort(ids2) # sort by ids
    max_diff = 0

    for i in range(len(sort1)):
        p1 = T1[sort1[i]]
        p2 = T2[sort2[i]]
        diff = np.abs(1-(p1/p2))
        max_diff = max(max_diff, diff)

    return max_diff
