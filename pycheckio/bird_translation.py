VOWELS = "aeiouy"


def translate(phrase):
    build_phrase = ''

    i = 0
    while i < len(phrase):
        if phrase[i] is ' ':
            build_phrase += phrase[i]
            i += 1
            continue
        if phrase[i] not in VOWELS:
            # remove next letter
            build_phrase += phrase[i]
            i += 2
            continue
        else:
            # remove next 2 letters
            build_phrase += phrase[i]
            i += 3

    print( "translation: {}".format(build_phrase))
    return build_phrase


if __name__ == '__main__':
    print("Example:")
    print(translate("hieeelalaooo"))

    # These "asserts" using only for self-checking and not necessary for auto-testing
    assert translate("hieeelalaooo") == "hello", "Hi!"
    assert translate("hoooowe yyyooouuu duoooiiine") == "how you doin", "Joey?"
    assert translate("aaa bo cy da eee fe") == "a b c d e f", "Alphabet"
    assert translate("sooooso aaaaaaaaa") == "sos aaa", "Mayday, mayday"
    print("Coding complete? Click 'Check' to review your tests and earn cool rewards!")
