from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfpage import PDFTextExtractionNotAllowed


def parse(path, Save_name):
    Path = open(path, "rb")
    parser = PDFParser(Path)
    document = PDFDocument(parser)

    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            for x in layout:
                if (isinstance(x, LTTextBoxHorizontal)):
                    with open('%s' % (Save_name), 'a') as f:
                        results = x.get_text().encode('utf-8')
                        f.write(str(results,encoding='utf-8')+ '\n')
def getAbstract(path):
    with open(path,'r') as f:
        lines=f.readlines()
        abstract=""
        start=False
        for line in lines:
            if line[:-1]=='Abstract':
                start=True
                continue
            if line[:-1]=="1 Introduction":
                break
            if start:
                abstract+=line
        return abstract
def getIntroduction(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        intro = ""
        start = False
        for line in lines:
            if line[:-1] == '1 Introduction':
                start = True
                continue
            if len(line)>2 and  line[:2] == "2 ":
                break
            if start:
                intro += line
        return intro
def handleParagraph(para):
    return para.replace("-\n","").replace("\n"," ")

teststr=getAbstract('test.txt')
print(teststr)
print((handleParagraph(teststr)))