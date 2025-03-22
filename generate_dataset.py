import json
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
import nltk
nltk.data.path.append('/nfs/hpc/share/baartmar/NSM/nltk_data')
#nltk.download('wordnet', download_dir='/nfs/hpc/share/baartmar/NSM/nltk_data')
#nltk.download('brown', download_dir='/nfs/hpc/share/baartmar/NSM/nltk_data')
#nltk.download('stopwords', download_dir='/nfs/hpc/share/baartmar/NSM/nltk_data')
#nltk.download('punkt_tab', download_dir='/nfs/hpc/share/baartmar/NSM/nltk_data')
#nltk.download('reuters')
# nltk.download('averaged_perceptron_tagger_eng', download_dir='/nfs/hpc/share/baartmar/NSM/nltk_data')
from nltk.corpus import wordnet, brown, stopwords
from nltk import pos_tag
import random
from collections import Counter
import string
import re
from tqdm import tqdm
import concurrent.futures

load_dotenv()

# load the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_PERSONAL_API_KEY"))

# load the NSM prompts from the json file
with open("nsm_prompt.json", "r", encoding="utf-8") as file:
    messages=json.load(file)

# define the NSM primes to be excluded from the dataset
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

# get the senses of a word from WordNet
def get_senses(word, exampler='gpt-4o-mini', max_senses=2):
    senses = []
    
    # Fetch senses from WordNet
    synsets = wordnet.synsets(word)
    
    # If there are more than max_senses, select a random subset
    if len(synsets) >= max_senses:
        synsets = random.sample(synsets, max_senses)
    
    # For each set of senses (synset), get examples and return them
    for syn in synsets:
        examples = []
        
        count = random.randint(1, 3)
        # Try to get examples from WordNet
        for example in syn.examples():
            if word.lower() in example.lower():
                examples.append({
                    "sentence": example,
                    "source": "WordNet"
                })
        
        # If WordNet doesn't provide enough examples, generate more using OpenAI API
        if len(examples) < count:
            remaining = count - len(examples) # how many more examples are needed
            example_prompt = [{
                "role": "user",
                "content": f"Word: '{word}'\nSense: {syn.definition()}\nPlease generate {remaining} example sentences using the word in this sentence. Only output the sentences, starting a new line for each example."
            }]

            # generate more examples via OpenAI API query
            response = client.chat.completions.create(
                model=exampler,
                messages=example_prompt,
                stream=False
            )

            # add the generated examples to the list of examples
            generated_examples = response.choices[0].message.content.strip().split('\n')
            for generated_example in generated_examples:
                examples.append({
                    "sentence": generated_example,
                    "source": exampler
                })
        
        # Add the sense and examples to the list
        senses.append({
            "definition": syn.definition(),
            "examples": examples
        })
    
    return senses

# generates dataset of words with their senses and explications
def generate_dataset(word_list, explications_per_word=2, explicator='gpt-4o', exampler='gpt-4o-mini'):
    dataset = {"data": [], "model": explicator}
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"data/data_{timestamp}.json"
    
    try:
        # Iterate over all words with a progress bar
        for word in tqdm(word_list, desc="Processing words", unit="word"):
            # Get senses and examples for the current word
            senses = get_senses(word, exampler)
            
            word_data = {
                "word": word,
                "senses": []
            }
            
            # Iterate over the senses with a progress bar
            for sense in tqdm(senses, desc=f"Processing senses for {word}", unit="sense", leave=False):
                passages = "\n".join([ex["sentence"] for ex in sense["examples"]])
                
                # Query GPT-4o for explication of the word sense
                explication_prompt = [{
                    "role": "user",
                    "content": f"Word/Phrase: {word}\nSense: {sense['definition']}\nExample Passages:\n{passages}\nExplication:"
                }]
                
                # Initialize the list of responses
                responses = []
                
                for _ in range(explications_per_word):
                    response = client.chat.completions.create(
                        model=explicator,
                        messages=messages + explication_prompt,
                        stream=False
                    )
                    responses.append(response.choices[0].message.content)
                
                # Append the sense with its examples and generated responses
                word_data["senses"].append({
                    "definition": sense["definition"],
                    "examples": sense["examples"],  # Each example has a source metadata
                    "responses": responses
                })
            
            # Add word data to the dataset
            dataset["data"].append(word_data)

        # Write the complete dataset to the file after processing all words
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=4)
            f.flush()  # Ensure data is written to disk immediately

    except Exception as e:
        # Handle errors by saving the current state of the dataset
        print(f"Error occurred: {e}. Saving the current dataset...")
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=4)
            f.flush()
        print(f"Dataset saved at {filename} before the error occurred.")
        raise  # Re-raise the exception after saving

    return filename


# loads the Brown corpus, filters out stop words, non-alphabetic words, NSM primes, and tags
words = brown.words() 
stop_words = set(stopwords.words('english'))
non_alpha_pattern = re.compile('^[^a-zA-Z]+$')

# selects most common words using filtering
word_counts = Counter(words)
common_words_original = [word for word, count in word_counts.most_common(20000)
                         if word.lower() not in stop_words
                         and not non_alpha_pattern.match(word)
                         and word.lower() not in NSM_PRIMES
                         and len(word) > 1]

tagged_words = pos_tag(common_words_original) # tag the words with their part of speech
filtered_common_words = [word.lower() for word, tag in tagged_words if tag not in ['NNP', 'NNPS']]
filtered_common_words = filtered_common_words[400:]
print(filtered_common_words[:100])
dataset_file = generate_dataset(filtered_common_words[:10000])
print(f"Dataset saved to: {dataset_file}")