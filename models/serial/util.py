import numpy as np

def get_parent_update_condition(condition, threshold=0.1):
    if condition in ("auto", 'fractional_difference'):
        return max_fractional_difference_update_condition(threshold)
    raise NotImplementedError(f"Parent update condition {condition} not implemented.")


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

class max_fractional_difference_update_condition:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, vertex):
        diff =  max_fractional_difference(vertex._last_update_sent, vertex.get_external_T())
        return diff >= self.threshold

