
def max_val(d):
    """return max value and its key of dict"""
    v = list(d.values())[1:]
    acc = [val[3] for val in v]
    k = list(d.keys())[1:]
    max_v = max(acc)
    return k[acc.index(max_v)], max_v
