OPERATION_NAMES = ("conjunction", "disjunction", "implication", "exclusive", "equivalence")

def boolean(x, y, operation):

    op_funcs = (conjunction(x,y), disjunction(x,y), implication(x,y), exclusive(x,y), equivalence(x,y))
    operation_sel_dict = dict(zip(OPERATION_NAMES, op_funcs))

    return operation_sel_dict.get(operation)

def conjunction(x,y):
    if x and y:
        return True
    else:
        return False

def disjunction(x,y):
    if x or y:
        return True
    else:
        return False

def implication(x,y):
    if bool(x) is True:
        return bool(y)
    else:
        return True

def exclusive(x,y):
    if x and not y:
        return True
    elif y and not x:
        return True
    else:
        return False

def equivalence(x,y):
    if x == y:
        return True
    else:
        return False

if __name__ == '__main__':
    #These "asserts" using only for self-checking and not necessary for auto-testing
    assert boolean(1, 0, "conjunction") == 0, "and"
    assert boolean(1, 0, "disjunction") == 1, "or"
    assert boolean(1, 1, "implication") == 1, "material"
    assert boolean(1, 0, "implication") == 0, "material"
    assert boolean(0, 1, "exclusive") == 1, "xor"
    assert boolean(0, 1, "equivalence") == 0, "same?"
    print('All good! Go and check the mission.')
