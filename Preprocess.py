from nltk import tokenize
import re
import random

# set seed
random.seed(42)


def extractor(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def sent_tokenize(path):
    text = extractor(path)
    return tokenize.sent_tokenize(text)


def string_stripper(s):
    s = s.lower()
    s = re.sub(r'[^\w\s\d+]', '', s)
    s = re.sub(r'[\n\r\t]', ' ', s)
    return s


def char_list_constructor(char_set, s):
    for char in s:
        if char.isalpha() and char not in char_set:
            char_set.append(char)

    return char_set


def sent_preprocess(path, char_lim, sample_num, lang, char_set):
    sent_list = sent_tokenize(path)
    pp_list = []
    for sentence in sent_list:
        if len(sentence) < char_lim:
            continue
        sentence = string_stripper(sentence)
        char_set = char_list_constructor(char_set, sentence)
        pp_list.append([sentence, lang])

    if len(pp_list) >= sample_num:
        pp_list = random.sample(pp_list, sample_num)

    return pp_list, char_set


