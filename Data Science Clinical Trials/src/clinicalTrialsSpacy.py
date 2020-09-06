# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 22:16:15 2020
The current code will extract Noun chunk, dependency parser and root text from the documnets
@author: Savitri
"""
# from langdetect import detect
import pandas as pd
import spacy

df = pd.read_csv('..\\data\\twitter.csv', encoding="utf-8")
print(df.info())
df.drop(['~', '~.1', '~.2'], axis=1)
import re


def remove_punct(x):
    x = re.sub(r'#', '', x)
    x = re.sub(r'https://\w+\.\w+/\w+', '', x)  # .encode('utf-8',errors='ignore')
    x = re.sub(r'\@\w+', '', x)
    x = re.sub(r'\d+', '', x)
    x = re.sub(r'\d+\.\d+', '', x)
    x = re.sub(r':', '', x)
    x = re.sub(r'!', '', x)
    x = re.sub(r'\n', ' ', x)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    x = emoji_pattern.sub(r'', x)
    x = x.replace('RT', '')
    x = x.strip()
    return x


# =============================================================================
# '''def languageDetect(x):
#     try:
#         print(len(x))
#         if len(x) >20:
#             print('text==',x)
#             return detect(x)
#         else:
#             print('text',x)
#             return x
# 
#     except NameError:
#         return detect(x)
#   
# 
# df['lang'] = df['textnew'].apply(languageDetect)
# #df.to_csv('D:\\TwitterResults\\testresultsDrug4.csv')'''
# =============================================================================


def nlpapply(x):
    nlp = spacy.load('en_core_web_sm')
    if len(x) > 20:
        doctext = nlp(x)
        print('doctext=', doctext)
        return doctext


import textacy


def findpattern(x):
    pattern = '<NOUN><ADP><PROPN>|<NOUN><CCONJ><NOUN>|<NOUN><PUNCT><NOUN>|<NOUN><NOUN>|<PROPN><AUX><VERB><NOUN>|<PROPN><ADP><NOUN>|<VERB><DET><NOUN>|<PROPN><VERB>|<NOUN><ADV><ADP><NOUN>|<ADV><ADJ><NOUN>|<NOUN><VERB>|<ADJ><NOUN>|<ADJ><PROPN>|<NOUN><VERB><NOUN>'  # '<NOUN>|<PROPN|<VERB>|<ADJ>'
    if list(textacy.extract.pos_regex_matches(x, pattern)):
        print('pattern=', list(textacy.extract.pos_regex_matches(x, pattern)))

        return list(textacy.extract.pos_regex_matches(x, pattern))


def find_dobj(tok):
    for token in list(tok):
        if token:
            if token.dep_ == 'dobj':
                print('dobj===', token.head.text,
                      str([child for child in token.children]).replace('[', '').replace(']', ''), token.text)
                return token.head.text, str([child for child in token.children]).replace('[', '').replace(']',
                                                                                                          ''), token.text


def find_root(doctext):
    for token in doctext:
        if token:
            if token.dep_ == 'ROOT':
                # print(token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children])
                print('Root=', token.head.text, [child for child in token.children])
                return token.head.text, str([child for child in token.children]).replace('[', '').replace(']', '')


def find_noun(doctext):
    for token in doctext:
        if token:
            if token.head.pos_ == 'NOUN':
                print('Noun=', token.head.text, token.text,
                      str([child for child in token.children]).replace('[', '').replace(']', ''))
                return str([child for child in token.children]).replace('[', '').replace(']',
                                                                                         ''), token.head.text, token.text


if __name__ == '__main__':
    # process the data in the batch of 10
    for k in range(0, 30, 10):
        start = int(k)
        end = start + 10
        try:
            df['textnew'] = df['text'][start:end].apply(remove_punct)
            df['processed'] = df['textnew'][start:end].apply(nlpapply)
            # df['processed'].dropna(inplace=True)
            df['terms'] = df['processed'][start:end].apply(findpattern)
            # df['terms'].dropna(inplace=True)
            df['DobjText'] = df['processed'][start:end].apply(find_dobj)
            df['RootText'] = df['processed'][start:end].apply(find_root)
            df['NounText'] = df['processed'][start:end].apply(find_noun)
            # df[start:end].to_csv('testresultsDrug'+str(k)+'.csv', index=False)
            print('----next------' + str(k))
        except:
            # pass
            print('none')
