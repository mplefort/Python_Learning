from typing import List


def checkio(game_result: List[str]) -> str:
    columns = ["".join(x) for x in zip(*game_result)]
    diagonals = ["".join(c) for c in zip(*[(x[i], x[2 - i]) for i, x in enumerate(game_result)])]

    combination = game_result + columns + diagonals

    if "XXX" in combination: return "X"
    if "OOO" in combination: return "O"
    return "D"

def check(game_result, p):

    win = False
    # horizontal case
    for row in game_result:
        count = 0
        for let in row:
            if let is p:
               count += 1
        if count == 3:
            return True

    # vertical case
    col1 = []
    col2 = []
    col3 = []
    for row in (game_result):
        col1 += row[0]
        col2 += row[1]
        col3 += row[2]
    cols = [col1, col2, col3]
    for col in cols:
        count = 0
        for let in col:
            if let is p:
                count += 1
        if count == 3:
            return True

    # diagonal cases
    diag1 = [game_result[0][0], game_result[1][1], game_result[2][2]]
    diag2 = [game_result[0][2], game_result[1][1], game_result[2][0]]
    diags = [diag1, diag2]
    for diag in diags:
        count = 0
        for let in diag:
            if let is p:
                count += 1
        if count == 3:
            return True

    return False

if __name__ == '__main__':
    print("Example:")
    print(checkio(["X.O",
                   "XX.",
                   "XOO"]))

    # These "asserts" using only for self-checking and not necessary for auto-testing
    assert checkio([
        "X.O",
        "XX.",
        "XOO"]) == "X", "Xs wins"
    assert checkio([
        "OO.",
        "XOX",
        "XOX"]) == "O", "Os wins"
    assert checkio([
        "OOX",
        "XXO",
        "OXX"]) == "D", "Draw"
    assert checkio([
        "O.X",
        "XX.",
        "XOO"]) == "X", "Xs wins again"
    print("Coding complete? Click 'Check' to review your tests and earn cool rewards!")