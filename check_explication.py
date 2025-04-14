import nltk
import os
from nltk.corpus import stopwords
import string
class LegalExplication:
    def __init__(self, original_word: str, explication: str, len_threshold: (int, int), non_prime_threshold: (int, int)):
        self.original_word = original_word
        self.explication = explication
        self.len_threshold = len_threshold
        self.non_prime_threshold = non_prime_threshold

    NSM_STOPWORDS = set();

    # class variable: NSM semantic primes
    NSM_PRIMES = {
        "I", "you", "someone", "people", "something", "thing", "body", "kind", "part",
        "this", "the same", "other", "else", "another", "one", "two", "some", "all",
        "much", "many", "little", "few", "good", "bad", "big", "small", "think", "know",
        "want", "don't want", "feel", "see", "hear", "say", "words", "true", "do",
        "happen", "move", "there", "is", "be",
        "mine", "live", "die", "when", "time", "now", "before", "after",
        "a long time", "a short time", "for some time", "moment", "where", "place",
        "here", "above", "below", "far", "near", "side", "inside", "touch", "contact",
        "not", "maybe", "can", "because", "if", "very", "more", "like", "as", "way"
    }

    def load_stopwords(self):
        """
        Load the stopwords from the NLTK stopwords corpus.
        """
        # Check if stopwords data exists in AppData/Roaming
        appdata_path = os.path.join(os.getenv('APPDATA'), 'nltk_data', 'corpora', 'stopwords')
        if not os.path.exists(appdata_path):
            nltk.download('stopwords')
        self.NSM_STOPWORDS = set(stopwords.words('english'))

    def is_prime(self, word: str) -> bool:
        """
        Check if a word is a prime word.
        """
        return word.lower() in self.NSM_PRIMES
    
    def is_non_prime_stop_grammar(self, word: str) -> bool:
        """
        Check if a word is a non-prime stop word or grammatical element.
        """
        return word.lower() in self.NSM_STOPWORDS
    
    def is_non_prime_molecule(self, word: str) -> bool:
        """
        Check if a word is a non-prime molecule.
        """
        return not self.is_prime(word) and not self.is_non_prime_stop_grammar(word)

    def check_legal_explication(self, original_word: str, explication: str, len_threshold: tuple[int, int], non_prime_threshold: tuple[int, int]):
        """
        Check if an explication meets validity criteria.

        Args:
            original_word (str): The word or phrase being explicated
            explication (str): The NSM explication text to validate
            len_threshold (tuple[int, int]): Min and max allowed length (number of words)
            non_prime_threshold (tuple[int, int]): Min and max allowed non-prime words

        Returns:
            length (int): Number of words in explication
            num_primes (int): Number of primes in explication
            num_non_prime_stop_grammar (int): Count of non-prime words classified as stop words or grammatical elements.
            num_non_prime_molecule (int): Count of non-prime non-stopword words classified as molecules.
            num_unique_molecules (int): Count of unique molecules in explication.
            contains_original_word (bool): Whether the explication contains the original word.
            uses_illegal_punctuation (bool): Whether the explication uses illegal punctuation.
            mol_to_word_ratio (float): Ratio of molecules to words in the explication.
        """

        # convert the explication to lowercase for uniformity
        explication = explication.lower()

        # Initialize flags
        uses_illegal_punctuation = False

        # Tokenize the explication
        tokens = explication.split()
        length = len(tokens)

        # Count primes and non-primes
        num_primes = sum(1 for token in tokens if self.is_prime(token))
        num_non_prime_stopwords = sum(1 for token in tokens if self.is_non_prime_stop_grammar(token))
        num_non_prime_molecules = sum(1 for token in tokens if self.is_non_prime_molecule(token))
        contains_original_word = original_word.lower() in explication
        # num_unique_molecules = len(set(tokens)) #TODO change this if it's for molecules only
        uses_illegal_punctuation = any(char in explication for char in string.punctuation if char not in {'.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}', '-'})
        mol_to_word_ratio = num_non_prime_molecules / length if length > 0 else 0
    
        return length, num_primes, num_non_prime_stopwords, num_non_prime_molecules, num_unique_molecules, contains_original_word, uses_illegal_punctuation, mol_to_word_ratio

    def annotate_explication(self, explication: str) -> str:
        """
        Prints an explication as an annotated string, where all the molecules are marked as [m]:
        """
        tokens = explication.split()
        annotated_tokens = []
        for token in tokens:
            if self.is_non_prime_molecule(token):
                annotated_tokens.append(f'{token} [m]')
            else:
                annotated_tokens.append(token)
        return ' '.join(annotated_tokens)
