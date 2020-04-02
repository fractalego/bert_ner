import os

from bert_ner.tagger import NERTagger

_path = os.path.dirname(__file__)

_filename = os.path.join(_path, '../data/ontonotes.model')

_text = """
John Stephen Smith is working for Acme at the Gherkin in London. 
"""

if __name__ == '__main__':
    tagger = NERTagger(_filename)
    predictions = tagger.tag(_text)
    print(predictions)
