from collections import defaultdict
from uuid import uuid4

def count_dict(depth=1):
    def _count_dict_inner(depth):
        if depth==1:
            return lambda : defaultdict(lambda : 0)
        return lambda : defaultdict(_count_dict_inner(depth-1))
    return _count_dict_inner(depth)()

inverse_dict = lambda d : dict(zip(d.values(), d.keys()))

get_unique_id = lambda : uuid4().int
