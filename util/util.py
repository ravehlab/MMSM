import sys
from collections import defaultdict
from uuid import uuid4
import numpy as np

def count_dict(depth=1):
    def _count_dict_inner(depth):
        if depth==1:
            return lambda : defaultdict(lambda : 0)
        return lambda : defaultdict(_count_dict_inner(depth-1))
    return _count_dict_inner(depth)()

inverse_dict = lambda d : dict(zip(d.values(), d.keys()))

get_unique_id = lambda : np.random.choice(sys.maxsize)
