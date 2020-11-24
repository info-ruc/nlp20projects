from nltk.corpus import wordnet as wn
print(wn.synsets('research') )

print(wn.synset('research.v.01').definition())
print(wn.synset('research.v.02').definition())
print(wn.synset('research.v.01').lemma_names())
print(wn.synset('research.v.02').lemma_names())