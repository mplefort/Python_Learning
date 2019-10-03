from codewars.test import Test

def find_it(seq):
    num_count = {}
    for idx, num in enumerate(seq):
        if num in num_count:
            continue
        else:
            num_count[num] = seq.count(num)

    for key, val in num_count.items():
        if val % 2 != 0:
            return int(key)

test = Test()
# test.describe("Example")
test.assert_equals(find_it([20,1,-1,2,-2,3,3,5,5,1,2,4,20,4,-1,-2,5]), 5)
