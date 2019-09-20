import unicodedata
import re

reg = re.compile(r' WITH .+?$')

def checkio(in_string):
    return ''.join(map(remove_accent, in_string))

def remove_accent(char):
    name = unicodedata.name(char)
    if name.startswith('COMBINING '):
        return ''
    name_wo_accent = reg.sub('', name)
    return unicodedata.lookup(name_wo_accent)

    # These "asserts" using only for self-checking and not necessary for auto-testing


if __name__ == '__main__':
    assert checkio(u"préfèrent") == u"preferent"
    assert checkio(u"loài trăn lớn") == u"loai tran lon"
    assert checkio(u"完好無缺") == u"完好無缺"
    print('Done')
