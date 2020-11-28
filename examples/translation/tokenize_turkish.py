import spacy
import sys
from trtokenizer.tr_tokenizer import SentenceTokenizer, WordTokenizer

word_tokenizer_object = WordTokenizer()

# nlp = spacy.load("xx_ent_wiki_sm")
# for line in sys.stdin:
#     doc = nlp.tokenizer(line)
#     for token in doc:
#         print(token.text.rstrip(), end=' ')
#     print('')

for line in sys.stdin:
    doc = word_tokenizer_object.tokenize(line)
    for token in doc:
        print(token.rstrip(), end=' ')
    print('')
