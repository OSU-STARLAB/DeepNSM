from utils import nltk, NSM_PRIMES, STOP_WORDS
#nltk.download("brown")
from nltk.corpus import wordnet, brown
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import json
import re
from collections import Counter
from tqdm import tqdm
import math
import statistics

C_MAX = 10
SCALE = 7500
EXPONENT = 2
PENALTY_EXPONENT = 1.5
DEPTH_THRESHOLD = 10.0 
DEPTH_WEIGHT = 10.0 
HYPO_BONUS_FACTOR = 2.0
MERO_BONUS_FACTOR = 1.0

def dynamic_candidates(rank):
    return max(3, math.floor(C_MAX / (1 + (rank / SCALE) ** EXPONENT)))

hyponym_cache = {}
def recursive_hyponym_count(syn):
    if syn.name() in hyponym_cache:
        return hyponym_cache[syn.name()]
    def helper(s, visited):
        total = 0
        for hypo in s.hyponyms():
            if hypo.name() not in visited:
                visited.add(hypo.name())
                total += 1 + helper(hypo, visited)
        return total
    count = helper(syn, set([syn.name()]))
    hyponym_cache[syn.name()] = count
    return count

meronym_cache = {}
def recursive_meronym_count(syn):
    if syn.name() in meronym_cache:
        return meronym_cache[syn.name()]
    def helper(s, visited):
        total = 0
        meros = s.part_meronyms() + s.substance_meronyms() + s.member_meronyms()
        for mero in meros:
            if mero.name() not in visited:
                visited.add(mero.name())
                total += 1 + helper(mero, visited)
        return total
    count = helper(syn, set([syn.name()]))
    meronym_cache[syn.name()] = count
    return count

word_sense_data = []
synset_degrees = Counter()

words = brown.words()
non_alpha_pattern = re.compile(r'^[^a-zA-Z]+$')
word_counts = Counter(words)

lemmatizer = WordNetLemmatizer()
common_words_original = [word for word, count in word_counts.most_common()
                         if word.lower() not in STOP_WORDS and not non_alpha_pattern.match(word) and len(word) > 1]
tagged_words = pos_tag(common_words_original)
filtered_common_words = list(dict.fromkeys(
    lemmatizer.lemmatize(word.lower()) for word, tag in tagged_words if tag not in ['NNP', 'NNPS']
))
filtered_common_words = [word for word in filtered_common_words if word not in NSM_PRIMES]

max_occurrence = max(word_counts.values())
primary_norms = [word_counts[word] / max_occurrence for word in filtered_common_words]
freq_threshold = statistics.median(primary_norms)

for syn in wordnet.all_synsets():
    related = (syn.hypernyms() + syn.instance_hypernyms() +
               syn.part_holonyms() + syn.substance_holonyms())
    for rel in related:
        synset_degrees[rel] += 1
        synset_degrees[syn] += 1

for word in tqdm(filtered_common_words, desc="Processing Words", unit="word"):
    synsets = wordnet.synsets(word)
    word_occurrence = word_counts.get(word, 0)
    norm_freq = word_occurrence / max_occurrence
    for idx, syn in enumerate(synsets):
        lemma_names = syn.lemma_names()
        is_proper = (syn.pos() == wordnet.NOUN and 
                     any(lemma[0].isupper() for lemma in lemma_names))
        if not is_proper:
            norm_occurrence = norm_freq if (idx == 0 and norm_freq >= freq_threshold) else 0
            degree = synset_degrees[syn]
            word_sense_data.append({
                "synset": syn.name(),
                "word": word,
                "definition": syn.definition(),
                "examples": [{"text": ex, "source": "WordNet"}
                             for ex in syn.examples() if word.lower() in ex.lower()],
                "degree": degree,
                "norm_occurrence": norm_occurrence
            })

word_sense_data.sort(key=lambda x: -x["degree"])
deg_rank = 0
for syn in word_sense_data:
    syn["degree_rank"] = deg_rank
    deg_rank += 1

word_sense_data.sort(key=lambda x: -x["norm_occurrence"])
occ_rank = 0
for syn in word_sense_data:
    syn["occ_rank"] = occ_rank
    syn["adjusted_occ_rank"] = occ_rank ** PENALTY_EXPONENT
    occ_rank += 1

for syn in word_sense_data:
    try:
        d = wordnet.synset(syn["synset"]).min_depth()
    except Exception:
        d = DEPTH_THRESHOLD
    if d < DEPTH_THRESHOLD:
        depth_penalty = (DEPTH_THRESHOLD - d) ** 2
    else:
        depth_penalty = 0
    hyponym_count = recursive_hyponym_count(wordnet.synset(syn["synset"]))
    meronym_count = recursive_meronym_count(wordnet.synset(syn["synset"]))
    relation_bonus = -HYPO_BONUS_FACTOR * math.log(1 + hyponym_count) - MERO_BONUS_FACTOR * math.log(1 + meronym_count)
    syn["score"] = 0.75 * syn["adjusted_occ_rank"] + 0.25 * syn["degree_rank"] + DEPTH_WEIGHT * depth_penalty + relation_bonus

word_sense_data.sort(key=lambda x: x["score"])
overall_rank = 0
for syn in word_sense_data:
    syn["overall_rank"] = overall_rank
    overall_rank += 1

total_candidates = 0
for syn in word_sense_data:
    cand = dynamic_candidates(syn["overall_rank"])
    syn["num_candidates"] = cand
    total_candidates += cand

for syn in word_sense_data:
    del syn["score"]
    del syn["adjusted_occ_rank"]
    del syn["occ_rank"]
    del syn["degree_rank"]
    del syn["degree"]
    del syn["norm_occurrence"]

# Save the updated data
with open("data/word_sense_data.json", "w") as f:
    json.dump(word_sense_data, f, indent=4)

print(f"{len(word_sense_data)} total senses collected with dynamic candidate allocation.")
print(total_candidates)


# We need a ranking of how "ordinary" a word sense is
# We want to leverage both occurrence data (from brown, or potentially another corpus since brown is kind of old)
# as well as the structured data in wordnet (synset relations (hypernym and holonym))

# We need to end up with a better ranking of how "ordinary" a word sense is. The signals are:
# - The word appears a lot in text corpus
# - It is part of synset that we can predict from the relational data in wordnet is very fundamental (maybe it has high indegree or certain relationships, etc.)

# What information to take into account
# - We need to come up with how to do this and properly weigh different data
# - You need to look into WordNet, how its structured, get an idea of what types of relationships are defined (use their docs and ChatGPT)
# - There may also be graph algorithms that can be applied to search the graph for addtl information (Ask ChatGPT if it has ideas for this)
# - Maybe we can just come up with a better weighting of the formula I'm already applying.

# Ultimately, you will need to check your solution by looking at the JSON file and using the eye test to determine whether:
# - very obscure senses of common words are being ranked too high
# - very low-occuring words with high "fundamentality" according to wordnet relational data are being ranked too high
# - the top ranks should be highly occurring, highly "fundamental" or ordinary
# - the lowest ranks should be rarely occurring, not very "fundamental" or ordinary
# - one strategy could be finding a set of words that correspond to the previously mentioned properties, ranking them yourself, and optimizing for weights that get their relative ranking the same as what you rank.

