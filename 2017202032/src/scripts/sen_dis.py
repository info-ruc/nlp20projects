import logging
import sys
import os
import re
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

input_file = 'abstract.txt'
'''
model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=0, min_count=5, workers=8)
model.save(input_file + '.model')
model.save_word2vec_format(input_file + '.vec')'''

sent_file = 'abstract_test.txt'
model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
model.save_sent2vec_format(sent_file + '.vec')

def get_dis(a,b):
    dist=0
    for i in range(len(a)):
        dist+=(eval(a[i])-eval(b[i]))**2
    return dist**(0.5)

evictsymbols=["[",']','%']
def isvalid(s):
    for symbol in evictsymbols:
        if symbol in s:
            return False
    return True
def synonyms_replace(s):
    fdic=open("dic.txt","r")
    content=fdic.readlines()
    dic={}
    for line in content:
        line=line.split()
        dic[line[0]]=line[1]
    for word in s:
        if word in dic.keys():
            s=s.replace(word,dic[word])
    return s

f=open("abstract_sentence.txt.vec","r",encoding='utf-8')
veclines=f.readlines()[1:]
f.close()

freader=open(sent_file+'.vec',"r",encoding='utf-8')
testlines=freader.readlines()
freader.close()
vec=testlines[1].split()[1:]

dic={}
for i in range(len(veclines)):
    vecline=veclines[i]
    sent_index = int(vecline.split()[0][5:])
    vecline = vecline.split()[1:]
    curdis = get_dis(vecline, vec)
    dic[sent_index]=curdis

lt=sorted(dic.items(),key= lambda kv:(kv[1],kv[0]))

f=open("abstract_sentence.txt","r",encoding='utf-8')
sentences=f.readlines()
f.close()

fwrite=open("test_result.txt","w+",encoding='utf-8')
count=0
for it in lt:
    if not isvalid(sentences[it[0]]):
        continue
    if count>=8:
        break
    count+=1
    fwrite.write(synonyms_replace(sentences[it[0]][:-1]))
    print(sentences[it[0]],end="")

fwrite.close()
program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)