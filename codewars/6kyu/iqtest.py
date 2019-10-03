from codewars.test import Test

def iq_test(numbers):
    nums = [int(intnum) for intnum in numbers.split(" ")]
    even_nums = []
    odd_nums = []
    for num in nums:
        if num % 2 == 0:
            even_nums.append(num)
        else:
            odd_nums.append(num)

        if len(even_nums) >= 2:
            for find_odd in nums:
                if find_odd % 2 != 0:
                    return nums.index(find_odd) + 1

        if len(odd_nums) >= 2:
            for find_even in nums:
                if find_even % 2 == 0:
                    return nums.index(find_even) + 1



Test = Test()

Test.assert_equals(iq_test("2 4 7 8 10"), 3)
Test.assert_equals(iq_test("1 2 2"), 1)