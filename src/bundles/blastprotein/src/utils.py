from itertools import count

class InstanceGenerator:
    _instance_iterator = count(1)
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return next(InstanceGenerator._instance_iterator)

def make_instance_name(prefix="bp"):
    return "".join([prefix, str(next(InstanceGenerator()))])
