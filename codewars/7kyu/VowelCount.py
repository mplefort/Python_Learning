
from string import ascii_lowercase
from string import ascii_uppercase

vowels = ["A", "E", "I", "O", "U"]
low_vowels = []
for let in vowels:
    low_vowels.append(ascii_lowercase[ascii_uppercase.index(let)])

vowels.extend(low_vowels)

def getCount(inputStr):
    num_vowels = 0
    for vowel in vowels:
        if vowel in inputStr:
            num_vowels += inputStr.count(vowel)

    return num_vowels



assert(getCount("abracadabrae")) ==  6