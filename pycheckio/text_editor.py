from typing import List


class Text:
    def __init__(self):
        self.font = None
        self.text = ""

    def set_font(self, font):
        self.font = font

    def write(self, s: str):
        self.text += s

    def show(self):
        if self.font:
            font = "[%s]" % self.font
        else:
            font = ""
        return "%s%s%s" % (font, self.text, font)

    def restore(self, version: List[str]):
        self.font = version[0]
        self.text = version[1]


class SavedText:
    def __init__(self):
        self.versions = []

    def save_text(self, obj: Text):
        self.versions.append([obj.font, obj.text])

    def get_version(self, n: int):
        return self.versions[n]
# import re
# import copy
#
# class Text:
#
#     def __init__(self):
#         self.text = ''
#         self.version = 0
#         self.font_set = False
#         self.font = ''
#
#     def write(self, text):
#         if self.font_set == True:
#             font_search = re.search(r"\](.*)\[", self.text)
#             self.text = font_search.group(1)
#             self.text += text
#             self.font_set = False
#             self.set_font(self.font)
#         else:
#             self.text += text
#         return
#
#     def set_font(self, font):
#         self.font = font
#         if self.font_set == False:
#             self.font_set = True
#             self.text = "[" + font + "]" + self.text + "[" + font + "]"
#         else:
#             # remove old font
#             font_search = re.search(r"\](.*)\[", self.text)
#             self.text = font_search.group(1)
#             self.text = "[" + font + "]" + self.text + "[" + font + "]"
#
#
#
#     def show(self):
#         return self.text
#
#     def restore(self, restored_state):
#         # restored_state = self.saved_text.get_version(version)
#         self.text = restored_state.text
#         self.version = restored_state.version
#         self.font_set = restored_state.font_set
#         self.font = restored_state.font
#
# class SavedText:
#     def __init__(self):
#         self.state = {}
#         self.version = 0
#
#
#     def save_text(self, Text):
#         self.state[self.version] = copy.deepcopy(Text)
#         self.version += 1
#
#     def get_version(self, number):
#         return self.state[number]
#

if __name__ == '__main__':
    # These "asserts" using only for self-checking and not necessary for auto-testing

    text = Text()
    saver = SavedText()

    text.write("At the very beginning ")
    saver.save_text(text)
    text.set_font("Arial")
    saver.save_text(text)
    text.write("there was nothing.")

    assert text.show() == "[Arial]At the very beginning there was nothing.[Arial]"

    text.restore(saver.get_version(0))
    assert text.show() == "At the very beginning "

