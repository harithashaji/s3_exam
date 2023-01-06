import nltk
new = "The big cat ate the little mouse who has after the fresh cheese"
new_tokens = nltk.word_tokenize(new)
print(new_tokens)
new_tag = nltk.pos_tag(new_tokens)
print(new_tag)
grammar = "NP:{<DT>,<JJ>,<NN>}"
chunkParser = nltk.word_tokenize('grammar')
chunked = chunkParser.parse('new_tag')
print(chunked)
chunked.draw()
