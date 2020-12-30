from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever



if __name__ == '__main__':
    document_store = InMemoryDocumentStore()

    # Let's first get some documents that we want to query
    # Here: 517 Wikipedia articles for Game of Thrones
    doc_dir = "data"
    #s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    #fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
    # It must take a str as input, and return a str.
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # We now have a list of dictionaries that we can write to our document store.
    # If your texts come from a different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries yourself.
    # The default format here is: {"name": "<some-document-name>, "text": "<the-actual-text>"}

    # Let's have a look at the first 3 entries:
    print(dicts[:3])
    # Now, let's write the docs to our DB.
    document_store.write_documents(dicts)


    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, context_window_size=500,batch_size=1)
    #reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="what determines the cost and performance of alocal network", top_k_retriever=3, top_k_reader=3)
    print_answers(prediction, details="minimal")