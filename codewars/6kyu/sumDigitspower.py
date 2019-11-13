import unittest

def sum_dig_pow(a, b): # range(a, b + 1) will be studied by the function

    out = []
    for num in range(a,b+1):
        str_num = str(num)
        sumofpow = 0
        for idx, char in enumerate(str_num):
            sumofpow += int(char) ** (idx + 1)
        if sumofpow == num:
            out.append(num)
    return out


test = unittest.case.TestCase()

# test.assertEqual( sum_dig_pow(1, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9])
# test.assertEqual(sum_dig_pow(1, 100), [1, 2, 3, 4, 5, 6, 7, 8, 9, 89])
# test.assertEqual(sum_dig_pow(10, 89),  [89])
# test.assertEqual(sum_dig_pow(10, 100),  [89])
# test.assertEqual(sum_dig_pow(90, 100), [])
test.assertEqual(sum_dig_pow(1676, 1676), [1676])