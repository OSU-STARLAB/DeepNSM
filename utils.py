import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger_eng')  
#nltk.download('brown')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Define NSM primes
NSM_PRIMES = {
    "i", "you", "someone", "people", "something", "thing", "body", "kind", "part",
    "this", "the same", "other", "else", "another", "one", "two", "some", "all",
    "much", "many", "little", "few", "good", "bad", "big", "small", "think", "know",
    "want", "don't want", "feel", "see", "hear", "say", "words", "true", "do",
    "happen", "move", "there", "is", "be",
    "mine", "live", "die", "when", "time", "now", "before", "after",
    "a long time", "a short time", "for some time", "moment", "where", "place",
    "here", "above", "below", "far", "near", "side", "inside", "touch",
    "not", "maybe", "can", "because", "if", "very", "more", "like", "as", "way", "said"
}

# Find stopwords that are not in NSM_PRIMES
STOP_WORDS = stop_words - NSM_PRIMES

LEGAL_PUNCTUATION = {"'", ".", ",", ":", "!", "?", "\"", "\n", "\t", "(", ")", "/" }

