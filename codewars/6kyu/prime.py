import time
import math

def is_prime(num):

    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    max_div = math.floor(math.sqrt(num))
    for i in range(3,1 + max_div, 2):
        if num % i == 0:
            return False
    return True


t0 = time.time()
c = 0

for n in range(1,20000):
    x = is_prime(n)
    c += x

print("total prime: {}".format(c))

t1 = time.time()
print("time {}".format(t1-t0))

