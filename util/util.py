from collections import defaultdict

def count_dict(depth=1): #TODO debug this
    dic = lambda : defaultdict(lambda : 0)
    for i in range(depth-1):
        dic = lambda : defaultdict(dic)
    return dic()

inverse_dict d = dict(zip(d.values(), d.keys()))

