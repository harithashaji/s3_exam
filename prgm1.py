import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
stop_words=set('sequence=nltk.word_tokenize("english")')
txt="Sanjana,rajiv and maya are my best friends"\
    "maya getting married in next year"\
    "marriage is big step in one's life"\
    "it is exciting and frightening"\
    "but friendship is a sacred bond between people" \
    "it is a special kind of love of us" \
    "many of you are searching for those friends" \
    "but not found the right one"
tokenized=sent_tokenize(txt)
for i in tokenized:
    wordLists=nltk.word_tokenize(i)
    wordList=[w for w in wordLists if not w in stop_words]
    tagging=nltk.pos_tag(stop_words)
    print(tagging)