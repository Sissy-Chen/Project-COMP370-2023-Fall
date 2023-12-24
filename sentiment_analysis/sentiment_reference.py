import nltk
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

sentence='\n\n\n\n            .........   /n/n/n/n  !!@$%@$#%@ It was\' a really\' don\'t good day ........... --------------'

from nltk.tag import pos_tag
twt = nltk.TweetTokenizer()
token = twt.tokenize(sentence)
print(token)
after_tagging = pos_tag(token)
print (token)
print (after_tagging)
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

sentiment = 0.0
tokens_count = 0

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

for word, tag in after_tagging:
    wn_tag = penn_to_wn(tag)
    print(word, wn_tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
        continue

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        continue

    synsets = wn.synsets(lemma, pos=wn_tag)
    if not synsets:
        continue

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    print(swn_synset)

    sentiment += swn_synset.pos_score() - swn_synset.neg_score()
    tokens_count += 1
print (sentiment)