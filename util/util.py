"""A collection of utility functions used by HierarchicalMSM, and related classes"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

import time
import sys
from collections import defaultdict
import scipy
import numpy as np

def count_dict(depth=1):
    depth = int(depth)
    if depth==1:
        return defaultdict(int)
    elif depth==2:
        return defaultdict(count_dict)
    else:
        raise ValueError("count_dict only supports depths 1 and 2.")

def count_dict_depracated(depth=1) -> defaultdict:
    """
    ##### DEPRACATED BECAUSE IT CAN'T BE PICKLED FOR MULTIPROCESSING #####
    Gets a defaultdict, in which the default value is 0.
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

get_unique_id = lambda : np.random.choice(2**32)


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

def sparse_matrix_from_count_dict(counts, ids):
    data = []
    indices = []
    indptr = []
    id_2_inx = {id:i for i, id in enumerate(ids)}
    for src in ids:
        indptr.append(len(data))
        for dest, count in counts[src].items():
            indices.append(id_2_inx[dest])
            data.append(count)
    indptr.append(len(data))
    return scipy.sparse.csr_matrix((data, indices, indptr))

def two_step_count_matrix(counts, ids):
    n = len(ids)
    id_2_inx = {id:i for i, id in enumerate(ids)}
    shape = (n*n, n)
    data = []
    rows = []
    cols = []

    for state in counts.keys():
        j = id_2_inx[state]
        for (prev_prev_state, prev_state), c in counts[state].items():
            i = id_2_inx[prev_prev_state]*n + id_2_inx[prev_state]
            data.append(c)
            rows.append(i)
            cols.append(j)
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsc()




