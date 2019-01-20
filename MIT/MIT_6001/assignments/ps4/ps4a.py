# Problem Set 4A
# Name: <your name here>
# Collaborators:
# Time Spent: x:xx test change

def add_item(item, sequence):

    sequence_copy = sequence[:]
    for i, c in enumerate(sequence_copy):
        # remove string before adding all its permutations with new letter
        sequence.remove(c)
        for j, d in enumerate(c):
            sequence.append(c[0:j] + item + c[j:])
            sequence.append(c + item)

    return sequence

def get_permutations(sequence):
    '''
    Enumerate all permutations of a given string a

    sequence (string): an arbitrary string to permute. Assume that it is a
    non-empty string.

    You MUST use recursion for this part. Non-recursive solutions will not be
    accepted.

    Returns: a list of all permutations of sequence

    Example:
    >>> get_permutations('abc')
    ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

    Note: depending on your implementation, you may return the permutations in
    a different order than what is listed here.
    '''

    seq_list = list(sequence)

    if len(sequence) == 1:
        return [sequence]

    else:
        per_list = []
        for i in range(len(seq_list)):
            let = seq_list[i]
            sub_seq = seq_list[:i] + seq_list[i+1:]
            for p in get_permutations(sub_seq):
                per_list.append([let] + p)
        return per_list


if __name__ == '__main__':
    # EXAMPLE
    print('test')
    example_input = 'abc'
    print('Input:', example_input)
    print('Expected Output:', ['abc', 'acb', 'bac', 'bca', 'cab', 'cba'])
    perm_list = get_permutations(example_input)
    perm_str = []
    for perm in perm_list:
        perm_str.append(''.join(perm))

    print('Actual Output:', perm_str)
