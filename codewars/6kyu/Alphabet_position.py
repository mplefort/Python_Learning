from random import randint
from string import ascii_lowercase
from string import ascii_uppercase

def alphabet_position(text):
    let_pos = ""
    for let in text:
        if let in ascii_uppercase:
            let_pos += " "
            let_pos +=  str(ascii_uppercase.index(let) + 1)
            continue
        elif let in ascii_lowercase:
            let_pos += " "
            let_pos +=  str(ascii_lowercase.index(let) + 1)
            continue

    if let_pos:
        let_pos = let_pos[1:]
    print(let_pos)
    print("20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11")
    return let_pos


assert alphabet_position("The sunset sets at twelve o' clock.") == "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11"
assert alphabet_position("The narwhal bacons at midnight.") == "20 8 5 14 1 18 23 8 1 12 2 1 3 15 14 19 1 20 13 9 4 14 9 7 8 20"

number_test = ""
for item in range(10):
    number_test += str(randint(1, 9))
test.assert_equals(alphabet_position(number_test), "")