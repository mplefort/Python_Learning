def checkio(number):
    n = number
    pigeons = 0
    pigeons_fed = []
    min = 1
    while True:
        pigeons += min
        if len(pigeons_fed) < pigeons:
            pigeons_fed.extend([0]*min)
        for pigeon in range(pigeons):
            pigeons_fed[pigeon] += 1
            n -= 1
            if n <= 0:
                num_pigeons_fed = len(list(filter(lambda a: a != 0, pigeons_fed)))
                return num_pigeons_fed
        min += 1


if __name__ == '__main__':
    #These "asserts" using only for self-checking and not necessary for auto-testing
    assert checkio(1) == 1, "1st example"
    assert checkio(2) == 1, "2nd example"
    assert checkio(5) == 3, "3rd example"
    assert checkio(10) == 6, "4th example"