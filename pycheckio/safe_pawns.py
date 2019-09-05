from string import ascii_lowercase

asciilist = list(ascii_lowercase)

def safe_pawns(pawns: set) -> int:

    # for pawn in pawns:
    #     leter = pawn[0]
    #     num = pawn[1]
    #     if letter-1 in pawns and pawns[letter-1] is num - 1:
    #          pawn = safe

    safe_pawn = 0
    for pawn in pawns:
        letter = pawn[0]
        num = int(pawn[1])
        if letter is not 'a':
            safe_letters = [asciilist[asciilist.index(letter) + 1], asciilist[asciilist.index(letter) - 1] ]
        else:
            safe_letters = [asciilist[asciilist.index(letter) + 1] ]

        for safe_letter in safe_letters:
            if safe_letter + str(num-1) in pawns:
                safe_pawn += 1
                break


    return safe_pawn


if __name__ == '__main__':
    # These "asserts" using only for self-checking and not necessary for auto-testing
    assert safe_pawns({"b4", "d4", "f4", "c3", "e3", "g5", "d2"}) == 6
    assert safe_pawns({"b4", "c4", "d4", "e4", "f4", "g4", "e5"}) == 1
    print("Coding complete? Click 'Check' to review your tests and earn cool rewards!")