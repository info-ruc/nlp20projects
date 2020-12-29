from nltk.tokenize import sent_tokenize

def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list

freader=open("abstract.txt","r",encoding='utf-8')
fwriter=open("abstrace_sentence.txt","w+",encoding='utf-8')

for line in freader.readlines():
    for sentence in sentence_token_nltk(line):
        if sentence in ["1"]:
            continue
        fwriter.write(sentence+"\n")

freader.close()
fwriter.close()