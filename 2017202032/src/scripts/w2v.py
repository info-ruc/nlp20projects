import spacy
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs, labels, filename='fig'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.savefig(filename)

#text="Terrible microwave. doesn't heat the food yet the outside of the microwave gets hot. never had a microwave that gets hot on the outside. working with the supplier to replace it."
text='''The incessant beeping is enough to make you want to put your own head inside the microwave. Next to that, when you select a defrost time, it literally stops every 30 seconds to tell you to turn it over. and wont start ubnless you open and close the door then push start again I HATE THIS THING. Good thing it LOOKS NICE because the features, the beeping, and the UI is absolutely UNNERVING. I am writing this as some steaks are defrosting in it and the constant stops and beeping has just finally gotten me to the point where I had to come here and whine about it.'''
nlp=spacy.load('en_core_web_lg')
#print(nlp.vocab['perfect'].vector)
doc=nlp(text)
embedding=np.array([])
word_list=[]
for token in doc:
    if not(token.is_punct) and not(token.text in word_list):
        word_list.append(token.text)
print(word_list)
for word in word_list:
    embedding = np.append(embedding, nlp.vocab[word].vector)
print(embedding.shape)
embedding = embedding.reshape(len(word_list), -1)
print(embedding.shape)

tsne=TSNE()
low_dim_embedding = tsne.fit_transform(embedding)
plot_with_labels(low_dim_embedding, word_list,"fig2")