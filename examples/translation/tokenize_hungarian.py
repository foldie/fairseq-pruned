import spacy
import huspacy
import sys

nlp = huspacy.load()
for line in sys.stdin:
    doc = nlp.tokenizer(line)
    for token in doc:
        print(token.text.rstrip(), end=' ')
    print('')