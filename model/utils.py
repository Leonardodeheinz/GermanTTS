

# helpers

def exists(v):
    return v is not None

def default(v,d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num & den) == 0

def xnor(x,y):
    return not (x ^ y)

