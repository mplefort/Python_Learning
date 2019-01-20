# Problem Set 2, hangman.py
# Name: Matthew LeFort
# Collaborators:
# Time spent:

# Hangman Game
# -----------------------------------
# Helper code
# You don't need to understand this helper code,
# but you will have to know how to use the functions
# (so be sure to read the docstrings!)
import random
import string

WORDLIST_FILENAME = "words.txt"


def load_words():
    """
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist

wordlist = load_words()

def choose_word(wordlist):
    """
    wordlist (list): list of words (strings)
    
    Returns a word from wordlist at random
    """
    return random.choice(wordlist)

# end of helper code

# -----------------------------------



def is_word_guessed(secret_word, letters_guessed):
    '''
    secret_word: string, the word the user is guessing; assumes all letters are
      lowercase
    letters_guessed: list (of letters), which letters have been guessed so far;
      assumes that all letters are lowercase
    returns: boolean, True if all the letters of secret_word are in letters_guessed;
      False otherwise
    '''
    # FILL IN YOUR CODE HERE AND DELETE "pass"
    guess_result = False
    
    # force all lower case
    secret_word = secret_word.lower()
    for i in range(len(letters_guessed)):
        letters_guessed[i] = letters_guessed[i].lower()    

    
    # get list of all distinct letters in word
    chars_sec_word = []
    for char in secret_word:
        if char in chars_sec_word:
            continue
        else:
            chars_sec_word.append(char)
    
    # check if distinct letters in secret word are in letters_guessed
    for char in chars_sec_word:
        if char not in letters_guessed:
            guess_result = False
            break
        else:
            guess_result = True
        

    return guess_result



def get_guessed_word(secret_word, letters_guessed):
    '''
    secret_word: string, the word the user is guessing
    letters_guessed: list (of letters), which letters have been guessed so far
    returns: string, comprised of letters, underscores (_), and spaces that represents
      which letters in secret_word have been guessed so far.
    '''
     # force all lower case
    secret_word = secret_word.lower()
    for i in range(len(letters_guessed)):
        letters_guessed[i] = letters_guessed[i].lower()    

    partial_ans = ['_ ']*len(secret_word)

    # if letter_guessed contained in secret_word change _ to letter
    for guess_char in letters_guessed:
        i = 0
        while True:
            i = secret_word.find(guess_char, i) # find(substr, index start search)
            if i == -1:
                break
            partial_ans[i] = guess_char
            i = i + 1

    partial_ans = ''.join(partial_ans)
    return partial_ans
# Test for get_guessed_word()
# print(get_guessed_word('Word', ['w', 'O', 'd', 'd']) )


def get_available_letters(letters_guessed):
    '''
    letters_guessed: list (of letters), which letters have been guessed so far
    returns: string (of letters), comprised of letters that represents which letters have not
      yet been guessed.
    '''
    avail_lets = list(string.ascii_lowercase)

    for letter in letters_guessed:
        if letter in avail_lets:
            avail_lets.remove(letter)

    avail_lets = ('').join(avail_lets)
    return avail_lets

# print(get_available_letters(['a', 'd', 'z', 'e']))

def hangman(secret_word):
    '''
    secret_word: string, the secret word to guess.
    
    Starts up an interactive game of Hangman.
    
    * At the start of the game, let the user know how many 
      letters the secret_word contains and how many guesses s/he starts with.
      
    * The user should start with 6 guesses

    * Before each round, you should display to the user how many guesses
      s/he has left and the letters that the user has not yet guessed.
    
    * Ask the user to supply one guess per round. Remember to make
      sure that the user puts in a letter!
    
    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computer's word.

    * After each guess, you should display to the user the 
      partially guessed word so far.
    
    Follows the other limitations detailed in the problem write-up.
    '''

    '''
    Implement A:
    A. Game Arch:
        select secret word at random from words.txt
        initialize with 6 guesses remaining
        - Start of game print
            1. letters in secret word
            2. number of guesses remaining
        - keep track of:
            1. letters user guessed
            2. remaining possible letters
    '''

    # initialize game
    numGuesses = 10          # remaining guesses
    lettersGuessed = []     # all letters guessed so far
    uniqueLetters = []
    warningsLeft = 3      # 3 warnings given before game over


    print('Welcome to the game Hangman!')
    print('I am thinking of a word that is %d letters long.' % len(secret_word))

    vowels = ['a', 'e', 'i', 'o', 'u'];
    constanants = list(string.ascii_lowercase)
    for let in vowels:
        constanants.remove(let)
    

    '''
    Implement B:
    B: User-Computer interaction
        - before each guess print:
            a. num guesses remaining
            b. letters not yet guessed
        - ask for guess (check for one letter (non symbol) and make lowercase)
        - respond if correct or wrong letter in word
        - display secret word with letter/underscores
        - print (-) to separate
    '''

    while numGuesses != 0:

        print('-'*20 + '\n')

        # check if user won and word and exit
        if is_word_guessed(secret_word, lettersGuessed):
            print('Congrats! You won')
            score = numGuesses * len(uniqueLetters)
            print('Your score was: %d' % score)
            break


        # print guesses remaming and letters not yet guessed
        print('You have %d guesses left' % (numGuesses))
        print('Available letters: %s' % (get_available_letters(lettersGuessed)) )

        # input guess and confirm lowercase letter
        print('Please guess a single letter: ', end='')
        charGuess = str(input())

        if not charGuess.isalpha():
            warningsLeft -= 1#
            print('Oops! That is not a valid letter. You have %d warnings left.' % warningsLeft)
            if warningsLeft is 0: break
            continue

        if len(charGuess) > 1:
            warningsLeft -= 1
            if warningsLeft is 0: break
            print('Oops! only one letter at a time. You have %d warnings left.' % warningsLeft)
            continue

        lettersGuessed.append(charGuess.lower())

        #  if correct guess, print current guess,
        if (lettersGuessed[-1] in secret_word ) and (lettersGuessed[-1] not in lettersGuessed[:-1]):
            print('Good guess: ', end='')
            print(get_guessed_word(secret_word, lettersGuessed))
            uniqueLetters.append(lettersGuessed[-1])
            continue
        # else incorrect and reduce nmber of guesses
        else:
            print('Incorrect, that letter is not min my word: ', end='')
            print(get_guessed_word(secret_word, lettersGuessed))
            if lettersGuessed[-1] in vowels: numGuesses -= 2
            if lettersGuessed[-1] in constanants: numGuesses -=1



    print('-'*20 + '\n')
    # check of lost by guesses
    if numGuesses == 0:
        print('Sorry, you ran ot of guesses. The word was: ', end='')
        print(secret_word)


    # -----------------------------------



def match_with_gaps(my_word, other_word):
    '''
    my_word: string with _ characters, current guess of secret word
    other_word: string, regular English word
    returns: boolean, True if all the actual letters of my_word match the 
        corresponding letters of other_word, or the letter is the special symbol
        _ , and my_word and other_word are of the same length;
        False otherwise: 
    '''

    word = my_word.replace(' ', '')
    match = True
    lenMatch = True
    letMatch = True
    match_ = True

    revealedLets = []
    blankIdx = []

    # match in length
    if len(word) != len(other_word):
        lenMatch = False
    
    if lenMatch is False:
        match = False
        return match


    # match letters revealed match letters in other_word
    for idx, let in enumerate(word):
        if let.isalpha():
            if let != other_word[idx]:
                letMatch = False
                break
    if letMatch is False:
        match = False
        return match


    # letters revealed are not in blankspaces
    # match_ 
    for idx, let in enumerate(word):
        # lsit of revealed letters
        if let.isalpha() and let not in revealedLets: revealedLets.append(let)
        if let == '_': blankIdx.append(int(idx))

    # check if any revealed letters in _ index of other_word
    for let in revealedLets:
        # if found letter in other_words _ index, break
        if match_ == False: break 
        for idx in blankIdx:
            if let == other_word[idx]:
                match_ = False
                break

    if match_ is False: 
        match = False
        return match


    return match

# print(match_with_gaps('a_ _ le', 'aides'))


def show_possible_matches(my_word):
    '''
    my_word: string with _ characters, current guess of secret word
    returns: nothing, but should print out every word in wordlist that matches my_word

    '''

    matchedWords = []

    for word in wordlist:
        if match_with_gaps(my_word, word): matchedWords.append(word)

    for word in matchedWords:
        print(word + ' ', end='')

    print('\n')
# show_possible_matches('a_ _ le')



def hangman_with_hints(secret_word):
    '''
    secret_word: string, the secret word to guess.
    
    Starts up an interactive game of Hangman.
    
    * At the start of the game, let the user know how many 
      letters the secret_word contains and how many guesses s/he starts with.
      
    * The user should start with 6 guesses
    
    * Before each round, you should display to the user how many guesses
      s/he has left and the letters that the user has not yet guessed.
    
    * Ask the user to supply one guess per round. Make sure to check that the user guesses a letter
      
    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computer's word.

    * After each guess, you should display to the user the 
      partially guessed word so far.
      
    * If the guess is the symbol *, print out all words in wordlist that
      matches the current guessed word. 
    
    Follows the other limitations detailed in the problem write-up.
    '''

    '''
    Implement A:
    A. Game Arch:
        select secret word at random from words.txt
        initialize with 6 guesses remaining
        - Start of game print
            1. letters in secret word
            2. number of guesses remaining
        - keep track of:
            1. letters user guessed
            2. remaining possible letters
    '''

    # initialize game
    numGuesses = 10          # remaining guesses
    lettersGuessed = []     # all letters guessed so far
    uniqueLetters = []
    warningsLeft = 3      # 3 warnings given before game over


    print('Welcome to the game Hangman!')
    print('I am thinking of a word that is %d letters long.' % len(secret_word))

    vowels = ['a', 'e', 'i', 'o', 'u'];
    constanants = list(string.ascii_lowercase)
    for let in vowels:
        constanants.remove(let)
    

    '''
    Implement B:
    B: User-Computer interaction
        - before each guess print:
            a. num guesses remaining
            b. letters not yet guessed
        - ask for guess (check for one letter (non symbol) and make lowercase)
        - respond if correct or wrong letter in word
        - display secret word with letter/underscores
        - print (-) to separate
    '''

    while numGuesses != 0:

        print('-'*20 + '\n')

        # check if user won and word and exit
        if is_word_guessed(secret_word, lettersGuessed):
            print('Congrats! You won')
            score = numGuesses * len(uniqueLetters)
            print('Your score was: %d' % score)
            break


        # print guesses remaming and letters not yet guessed
        print('You have %d guesses left' % (numGuesses))
        print('Available letters: %s' % (get_available_letters(lettersGuessed)) )

        # input guess and confirm lowercase letter
        print('Please guess a single letter: ', end='')
        charGuess = str(input())

        if charGuess is '*':
            show_possible_matches(get_guessed_word(secret_word, lettersGuessed))
            continue


        if not charGuess.isalpha():
            warningsLeft -= 1
            print('Oops! That is not a valid letter. You have %d warnings left.' % warningsLeft)
            if warningsLeft is 0: break
            continue

        if len(charGuess) > 1:
            warningsLeft -= 1
            if warningsLeft is 0: break
            print('Oops! only one letter at a time. You have %d warnings left.' % warningsLeft)
            continue

        lettersGuessed.append(charGuess.lower())

        #  if correct guess, print current guess,
        if (lettersGuessed[-1] in secret_word ) and (lettersGuessed[-1] not in lettersGuessed[:-1]):
            print('Good guess: ', end='')
            print(get_guessed_word(secret_word, lettersGuessed))
            uniqueLetters.append(lettersGuessed[-1])
            continue
        # else incorrect and reduce nmber of guesses
        else:
            print('Incorrect, that letter is not min my word: ', end='')
            print(get_guessed_word(secret_word, lettersGuessed))
            if lettersGuessed[-1] in vowels: numGuesses -= 2
            if lettersGuessed[-1] in constanants: numGuesses -=1



    print('-'*20 + '\n')
    # check of lost by guesses
    if numGuesses == 0:
        print('Sorry, you ran ot of guesses. The word was: ', end='')
        print(secret_word)



# When you've completed your hangman_with_hint function, comment the two similar
# lines above that were used to run the hangman function, and then uncomment
# these two lines and run this file to test!
# Hint: You might want to pick your own secret_word while you're testing.


if __name__ == "__main__":
    # pass

    # Load the list of words into the variable wordlist
    # so that it can be accessed from anywhere in the program


    # To test part 2, comment out the pass line above and
    # uncomment the following two lines.
    
    # secret_word = choose_word(wordlist)
    # # print(secret_word)
    # hangman(secret_word)


###############
    
    # To test part 3 re-comment out the above lines and 
    # uncomment the following two lines. 
    
    secret_word = choose_word(wordlist)
    hangman_with_hints(secret_word)
