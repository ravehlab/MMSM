import time
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
    Default count_dict:
    >>> d = count_dict()
    >>> d['what is the default value?']
    0
    >>> d['can I add to the default value?'] += 3; print(d['can I add to the default value?'])
    3

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

def get_threshold_check_function(greater_than : dict={}, less_than : dict={}, max_time=None):
    """
    Get a function that returns True if keyword arguments are greater or less than certain values,
    or if a certain amount of time has passed since calling this function

    Parameters
    ----------
    greater_than : dict, optional
        A dictionary of maximum allowed values by keyword. If the returned function is called with a
        keyword in this dictionary, the function will return True if the value is greater than the
        value assigned to that keyword in greater_than.
    less_than : dict, optional
        A dictionary of minimum allowed values by keyword. If the returned function is called with a
        keyword in this dictionary, the function will return True if the value is less than the
        value assigned to that keyword in less_than.
    max_time : int or float, optional
        The maximum time in seconds before the returned function will always be evaluated to True

    Returns
    -------
    check_function : callable
        A function that accepts keyword arguments, and returns True if one or more of the arguments
        has a value greater than the value of that keyword in greater_than or less the the value of
        that keyword in less_than, or if longer than max_time seconds have passed since this
        function was called.


    Examples
    --------
    get a function that checks whether a>5 or b>1.2:
    >>> gt = {'a' : 5, 'b' : 1.2}
    >>> check = get_threshold_check_function(gt)
    >>> check(a=3)
    False
    >>> check(a=3, b=2)
    True
    >>> check()
    False
    >>> check(undefined_keyword=0)
    False
    >>> check(undefined_keywords_have_no_effect=0, b=2)
    True

    get a function that checks whether a>5, b<0 or 3.5 seconds have passed
    >>> gt = {'a' : 5}
    >>> lt = {'b' : 1.2}
    >>> check = get_threshold_check_function(less_than=lt, greater_than=gt, max_time=3.5)
    >>> check(a=3, b=3)
    True
    >>> check(a=3, b=0)
    False
    >>> sleep(3.5)
    >>> check(any_keyword_will_work=3)
    True
    >>> check()
    True
    """
    gt = defaultdict(lambda : np.inf) 
    gt.update(greater_than) # val > gt[key] will be avaluated to False if key is not in greater_than
    lt = defaultdict(lambda : - np.inf)
    lt.update(less_than) # val < lt[key] will be avaluated to False if key is not in less_than
    if max_time:
        start_time = time.time()

    def _check_function(**kwargs):
        for key, val in kwargs.items():
            if (val > gt[key]) or (val < lt[key]):
                return True
            if max_time:
                return time.time() > start_time + max_time
        return False
    return _check_function
