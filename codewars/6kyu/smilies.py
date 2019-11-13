from unittest import TestCase
import re

def count_smileys(arr):
    # :) :D ;-D :~)
    count = 0
    for pot_smile in arr:
        x = re.search(r"(:|;)(~|-)?(\)|D)", pot_smile)
        if x:
            count += 1
    return count

Test = TestCase()
Test.assertEqual(count_smileys([':D',':~)',';~D',':)']), 4)
Test.assertEqual(count_smileys([':)',':(',':D',':O',':;']), 2)
Test.assertEqual(count_smileys([';]', ':[', ';*', ':$', ';-D']), 1)

