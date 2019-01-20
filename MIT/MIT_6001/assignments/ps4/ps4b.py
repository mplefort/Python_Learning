# Problem Set 4B
# Name: <your name here>
# Collaborators:
# Time Spent: 3:30 - 4:00, 11:00 - 1:30

import string

UNIT_TEST = False

### HELPER CODE ###
def load_words(file_name):
    """
    file_name (string): the name of the file containing
    the list of words to load

    Returns: a list of valid words. Words are strings of lowercase letters.

    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print("Loading word list from file...")
    # inFile: file
    in_file = open(file_name, 'r')
    # wordlist: list of strings
    wordlist = []
    for line in in_file:
        wordlist.extend([word.lower() for word in line.split(' ')])
    print("  ", len(wordlist), "words loaded.")
    return wordlist


def is_word(word_list, word):
    """
    Determines if word is a valid word, ignoring
    capitalization and punctuation

    word_list (list): list of words in the dictionary.
    word (string): a possible word.

    Returns: True if word is in word_list, False otherwise

    Example:
    >>> is_word(word_list, 'bat') returns
    True
    >>> is_word(word_list, 'asdf') returns
    False
    """
    word = word.lower()
    word = word.strip(" !@#$%^&*()-_+={}[]|\:;'<>?,./\"")
    return word in word_list


def get_story_string():
    """
    Returns: a story in encrypted text.
    """
    f = open("story.txt", "r")
    story = str(f.read())
    f.close()
    return story


### END HELPER CODE ###
WORDLIST_FILENAME = 'words.txt'
WORD_LIST = load_words(WORDLIST_FILENAME)



class Message(object):
    def __init__(self, text: str):
        """
        Initializes a Message object

        text (string): the message's text

        a Message object has two attributes:
            self.message_text (string, determined by input text)
            self.valid_words (list, determined using helper function load_words)
        """

        try:
            assert isinstance(text, str)
        except:
            print('expected a message to be string')
            return

        self.message_text = text
        self.valid_words = self.extract_words(text)

        print('valid words checked')

    def extract_words(self, message):
        valid_words = []
        for word in message.split(' '):
            if is_word(WORD_LIST, word):
                valid_words.append(word)

        return valid_words


    def get_message_text(self):
        '''
        Used to safely access self.message_text outside of the class
        
        Returns: self.message_text
        '''
        return self.message_text


    def get_valid_words(self):
        '''
        Used to safely access a copy of self.valid_words outside of the class.
        This helps you avoid accidentally mutating class attributes.
        
        Returns: a COPY of self.valid_words
        '''
        valid_words_copy = self.valid_words[:]

        return valid_words_copy

    def build_shift_dict(self, shift: int):
        """
        Creates a dictionary that can be used to apply a cipher to a letter.
        The dictionary maps every uppercase and lowercase letter to a
        character shifted down the alphabet by the input shift. The dictionary
        should have 52 keys of all the uppercase letters and all the lowercase
        letters only.

        shift (integer): the amount by which to shift every letter of the
        alphabet. 0 <= shift < 26

        Returns: a dictionary mapping a letter (string) to
                 another letter (string).
                 i.e. {'X':'A'} with shift 3
        """

        lower_case_cipher = {}
        for i, let in enumerate(string.ascii_lowercase):
            shift_index = i + shift
            if shift_index > len(string.ascii_lowercase)-1:
                shift_index = shift_index - len(string.ascii_lowercase)
            lower_case_cipher[let] = string.ascii_lowercase[shift_index]

        upper_case_cipher = {}
        for i, let in enumerate(string.ascii_uppercase):
            shift_index = i + shift
            if shift_index > len(string.ascii_uppercase) - 1:
                shift_index = shift_index - len(string.ascii_uppercase)
            upper_case_cipher[let] = string.ascii_uppercase[shift_index]

        shift_dict = {}
        shift_dict.update(lower_case_cipher)
        shift_dict.update(upper_case_cipher)

        return shift_dict



    def apply_shift(self, shift):
        '''
        Applies the Caesar Cipher to self.message_text with the input shift.
        Creates a new string that is self.message_text shifted down the
        alphabet by some number of characters determined by the input shift        
        
        shift (integer): the shift with which to encrypt the message.
        0 <= shift < 26

        Returns: the message text (string) in which every character is shifted
             down the alphabet by the input shift
        '''

        shift_dict = self.build_shift_dict(shift)

        encrypt_mes = []
        for let in self.message_text:
            if let in shift_dict:
                encrypt_mes.append(shift_dict[let])
            else:
                encrypt_mes.append(let)

        encrpyt_str = ('').join(encrypt_mes)

        return encrpyt_str



class PlaintextMessage(Message):
    def __init__(self, text, shift):
        """
        Initializes a PlaintextMessage object

        text (string): the message's text
        shift (integer): the shift associated with this message

        A PlaintextMessage object inherits from Message and has five attributes:
            self.message_text (string, determined by input text)
            self.valid_words (list, determined using helper function load_words)
            self.shift (integer, determined by input shift)
            self.encryption_dict (dictionary, built using shift)
            self.message_text_encrypted (string, created using shift)

        SimpleVirus.__init__(self, maxBirthProb, clearProb)
        """
        # Creates self.meessage_text, self.valid_words
        Message.__init__(self, text)
        self.shift = shift
        self.encryption_dict = self.build_shift_dict(self.shift)
        self.message_text_encrypted = self.apply_shift(self.shift)



    def get_shift(self):
        """
        Used to safely access self.shift outside of the class

        Returns: self.shift
        """

        return self.shift

    def get_encryption_dict(self):
        """
        Used to safely access a copy self.encryption_dict outside of the class

        Returns: a COPY of self.encryption_dict
        """
        return self.encryption_dict[:]

    def get_message_text_encrypted(self):
        """
        Used to safely access self.message_text_encrypted outside of the class

        Returns: self.message_text_encrypted
        """
        return self.message_text_encrypted


    def change_shift(self, shift):
        """
        Changes self.shift of the PlaintextMessage and updates other
        attributes determined by shift.

        shift (integer): the new shift that should be associated with this message.
        0 <= shift < 26

        Returns: nothing
        """
        self.shift = shift
        # PlaintextMessage.__init__(self.message_text, self.shift)
        self.encryption_dict = self.build_shift_dict(self.shift)
        self.message_text_encrypted = self.apply_shift(self.shift)



class CiphertextMessage(PlaintextMessage):
    def __init__(self, text):
        """
        Initializes a CiphertextMessage object

        text (string): the message's text

        a CiphertextMessage object has two attributes:
            self.message_text (string, determined by input text)
            self.valid_words (list, determined using helper function load_words)
        """
        # Create self.message_text and self.valid_words
        PlaintextMessage.__init__(self, text, 0)

    def decrypt_message(self):
        """
        Decrypt self.message_text by trying every possible shift value
        and find the "best" one. We will define "best" as the shift that
        creates the maximum number of real words when we use apply_shift(shift)
        on the message text. If s is the original shift value used to encrypt
        the message, then we would expect 26 - s to be the best shift value
        for decrypting it.

        Note: if multiple shifts are equally good such that they all create
        the maximum number of valid words, you may choose any of those shifts
        (and their corresponding decrypted messages) to return

        Returns: a tuple of the best shift value used to decrypt the message
        and the decrypted message text using that shift value
        """
        words_found = 0
        shift = 0
        decrypt_message = ''
        best_shift = 0
        num_valid_words = 0

        for i in range(1, 27):
            shift = i
            self.change_shift(shift)

            words = self.extract_words(self.message_text_encrypted)
            if num_valid_words < len(words):
                num_valid_words = len(words)
                best_shift = shift
                decrypt_message = self.message_text_encrypted

            self.change_shift(26 - shift)

        best_shift = 26 - best_shift

        return (best_shift, decrypt_message)

if __name__ == '__main__':
    #    #Example test case (PlaintextMessage)
    #    plaintext = PlaintextMessage('hello', 2)
    #    print('Expected Output: jgnnq')
    #    print('Actual Output:', plaintext.get_message_text_encrypted())
    #
    #    #Example test case (CiphertextMessage)
    #    ciphertext = CiphertextMessage('jgnnq')
    #    print('Expected Output:', (24, 'hello'))
    #    print('Actual Output:', ciphertext.decrypt_message())

    # WRITE YOUR TEST CASES HERE
    if UNIT_TEST == True:
        #
        # # Test Message class
        # my_message = Message('This, is my secret message z!')  # type: Message
        #
        # print('\nTest get_valid_words method')
        # main_valid_words = my_message.get_valid_words()
        # print('expected: ', end='')
        # print(my_message.get_valid_words())
        # main_valid_words.append('newWord')                # append word to check if copy returns
        # print('Returns:  ', end='')
        # print(my_message.get_valid_words())
        #
        # print('\nTest get_message_text method')
        # print('expected: ', end='')
        # print(my_message.get_message_text())
        # main_my_message = my_message.get_message_text()    # change main scope my_message text
        # main_my_message = 'New string message'
        # print('Returns:  ', end='')
        # print(my_message.get_message_text())
        #
        # # Test build_shift_dict
        # print('\nTest build_shift_dict method')
        # test_shift_dict = my_message.build_shift_dict(5)
        # if test_shift_dict['A'] is 'F':
        #     print('Correct shift 5 from "A" to "F"')
        # if test_shift_dict['X'] == 'C':
        #     print('Correct shift 5 from "X" to "C"')
        #
        # # Test applying encryption to message
        # print('\n Test apply_shift of 1: ')
        # encryption = my_message.apply_shift(1)
        # print('Message = ' + my_message.get_message_text())
        # print('Encrypt = ' + encryption)

        # Test PlainTextMessage class
        plain_msg = PlaintextMessage('This, is my plain message', 1)

        # Test applying encryption to message
        print('\n Test apply_shift of 1: ')
        print('Message = ' + plain_msg.get_message_text())
        print('Encrypt = ' + plain_msg.message_text_encrypted)

        print('\n Test apply_shift of 2: ')
        plain_msg.change_shift(2)
        print('Message = ' + plain_msg.get_message_text())
        print('Encrypt = ' + plain_msg.message_text_encrypted)

        # Test CiphertextMessage class
        cipher_message = CiphertextMessage('Uijt, jt nz qmbjo nfttbhf')
        (best_shift, decrypt_message) = cipher_message.decrypt_message()
        print('Message = ' + cipher_message.get_message_text())
        print('Encrypt = {}, Shift applied: {}'.format(decrypt_message, best_shift))

    # TODO: best shift value and unencrypted story
    encrypted_story = get_story_string()
    cipher_story = CiphertextMessage(encrypted_story)
    (best_shift, decrypted_story) = cipher_story.decrypt_message()
    print(decrypted_story)





