def non_repeat(line):
    """
        the longest substring without repeating chars
    """
    max_string = []
    line_copy = line[:]
    for i, letter in enumerate(line):
        max_string.append('')

        for let in line_copy:
            if let not in max_string[i]:
                max_string[i] += let
            else:
                line_copy = line_copy[1:]
                break

    max = 0
    for string in max_string:
        if len(string) > max:
            maxest_string = string
            max = len(string)

    return maxest_string


if __name__ == '__main__':
    # These "asserts" using only for self-checking and not necessary for auto-testing
    assert non_repeat('aaaaa') == 'a', "First"
    assert non_repeat('abdjwawk') == 'abdjw', "Second"
    assert non_repeat('abcabcffab') == 'abcf', "Third"
    print('"Run" is good. How is "Check"?')