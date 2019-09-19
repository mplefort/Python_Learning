###########################
# 6.0002 Problem Set 1b: Space Change
# Name:
# Collaborators:
# Time:
# Author: charz, cdenise

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """

    weights = sorted(egg_weights, reverse=True)

    taken_eggs = 0

    for weight in weights:
        if target_weight // weight >= 1:
            taken_eggs += target_weight // weight
            target_weight = target_weight % weight

    return taken_eggs


# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 3, 4, 5, 8, 10, 15, 20, 25, 26)
    n = 200000
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()

    egg_weights = (1, 5, 10, 25)
    n = 25
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 25")
    print("Expected ouput: 1")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print("\n")

    egg_weights = (1, 5, 10, 25)
    n = 4
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 4")
    print("Expected ouput: 4")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print("\n")