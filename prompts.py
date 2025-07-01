import json
from google.genai import types
from enum import Enum, auto
import re
from utils import *
# System Instruction Templates
import spacy
nlp = spacy.load("en_core_web_sm")


WORD_EXAMPLE_SYS_INST = """Generate {num_examples} distinct example sentences using the specified word strictly according to its defined sense. Ensure each sentence appears on a new line, without using numerical or bullet point separators. Only the sentences should be provided as output."""

AMBIG_EXAMPLE_SYS_INST = """Generate {num_examples} distinct example paragraphs using the specified word according to its defined sense, ensuring the word is masked as [MASK].

- The word must only appear once in the paragraph.
- Each paragraph should be 3 sentences long.
- Ensure there is no use of numbers or bullets to separate paragraphs. Each should appear on a new line.
- Ensure the surrounding paragraph is highly ambiguous and could not be used to guess the masked word.

Only provide the paragraphs as the output."""


NSM_EXPLICATION_SYS_INST="""You are a linguist specializing in semantic analysis using the Natural Semantic Metalanguage (NSM) approach. NSM is a linguistic theory that reduces lexicons down to a set of universal semantic primes. You will be given a word and example passages where the word is used. Your task is to paraphrase the word's meaning using the NSM primes. Here are guidelines for your paraphrase:
- Use NSM primes and simple words from ordinary language.
- The paraphrase should be exhaustive and should portray the full meaning of the word being analyzed. 
- The paraphrase must be able to replace the original word in ALL of the examples without changing the meaning.
- Do not use the original word or a close synonym to it.
- Avoid obscurity, circularity, and do not introduce words that are more complex than the original. 
- Do not use logical symbols or abbreviations.
List of NSM Primes:
I, you, someone, people, something/thing, body, kind, part, the same, other/else/another, one, two, some, all, much/many, little/few, good, bad, big, small, think, know, want, don't want, feel, see, hear, say, words, true, do, happen, move, be (somewhere), there is, be (someone/something), (is) mine, live, die, when/time, now, before, after, a long time, a short time, for some time, moment, where/place, here, above, below, far, near, side, inside, touch (contact), not, maybe, can, because, if, very, more, like/as/way."""

RECOVERY_PROMPT_SYS_INST = """Read the passage with a missing word indicated by <UNK>. Predict the missing word in the passage. Output only your prediction, without any additional text."""

RECOVERY_PROMPT_EXPLICATION_SYS_INST = """Read the passage provided with a missing word indicated by <UNK>. You will also be given a paraphrase that describes the meaning of the missing word. Use this paraphrase to predict the missing word in the passage. Output only your prediction, without any additional text."""

# User Message Templates

EXPLICATION_USER_TEMPLATE = """Word: {word}
Examples:
{examples}
Paraphrase:
"""

EXAMPLE_USER_TEMPLATE = """Word: '{word},' used in the sense '{definition}'"""

RECOVERY_USER_TEMPLATE = """Passage: {passage}
Missing Word:"""

RECOVERY_USER_TEMPLATE_EXPLICATION = """Passage: {passage}
Paraphrase:
{paraphrase}
Missing word:"""


class Explication:
    def __init__(self, text:str):
        self.text = text
        self.target_word = ""
        self.length = 0
        self.primes = 0
        self.stop_words = 0
        self.molecules = 0
        self.unique_molecules = 0
        self.uses_original_word=False
        
        self.primes_ratio = 0.0
        self.molecules_ratio = 0.0

        # grader ambig [1,2] if min/ent
        self.sub_scores = []

        self.avg_delta = 0.0
        self.avg_delta_min = 0.0
        self.avg_delta_ent = 0.0

        self.score_exp = 0.0
        self.total_score = 0.0

        self.ct_comet_scores = []
        self.ct_bleu_scores = []
        self.ct_embed_scores = []

    def legality_score(self, word:str):
        tokens = re.sub(r'[^\w\s]', '', self.text.lower()).split()
        self.target_word = word
        self.length = len(tokens)
        self.primes = sum(1 for t in tokens if t in NSM_PRIMES)
        self.stop_words = sum(1 for t in tokens if t in STOP_WORDS)
        all_molecules = [t for t in tokens if t not in NSM_PRIMES and t not in STOP_WORDS]
        self.molecules = len(all_molecules)
        self.unique_molecules = len(set(all_molecules))
        # Lemmatize the input word
        word_lemma = nlp(word.lower())[0].lemma_
        # Lemmatize the tokens and check for a match
        doc = nlp(" ".join(tokens))
        self.uses_original_word = any([word_lemma == token.lemma_ for token in doc if not token.is_space])
        self.primes_ratio = self.primes / self.length if self.length > 0 else 0
        self.molecules_ratio = self.molecules / self.length if self.length > 0 else 0

    def calculate_averages(self):
        self.avg_delta = sum([score.avg_delta_log for score in self.sub_scores]) / len(self.sub_scores) if self.sub_scores else 0
        self.avg_delta_min = sum([score.avg_min_delta_log for score in self.sub_scores]) / len(self.sub_scores) if self.sub_scores else 0
        self.avg_delta_ent = sum([score.avg_ent_delta_log for score in self.sub_scores]) / len(self.sub_scores) if self.sub_scores else 0
        self.score_exp = sum([score.adj_score for score in self.sub_scores]) / len(self.sub_scores) if self.sub_scores else 0
        self.total_score = 2 * (self.score_exp + (10 * self.primes_ratio) - (10 * self.molecules_ratio)) if not self.uses_original_word else 0.0

    def get_truncated(self, max_lines_remove=2):
        truncated_exps = []
        lines = self.text.strip().split('\n')
        for i in range(min(len(lines), max_lines_remove)):
            truncated_exp = Explication('\n'.join(lines[:-(i+1)]))
            truncated_exps.append(truncated_exp)
        return truncated_exps

    def pretty_print(self):
        pass

    def __json__(self):
        return {
            "target_word": self.target_word,
            "text": self.text,
            "uses_original_word": self.uses_original_word,
            "total_score": self.total_score,
            "score_exp": self.score_exp,
            "primes_ratio": self.primes_ratio,
            "molecules_ratio": self.molecules_ratio,
            "comet_scores": self.ct_comet_scores,
            "bleu_scores": self.ct_bleu_scores,
            "embed_scores": self.ct_embed_scores,
            "avg_delta": self.avg_delta,
            "avg_delta_min": self.avg_delta_ent,
            "avg_delta_ent": self.avg_delta_min,
            "length": self.length,
            "primes": self.primes,
            "stop_words": self.stop_words,
            "molecules": self.molecules,
            "unique_molecules": self.unique_molecules,
            "sub_scores": [score.__json__() for score in self.sub_scores]
        }

class AmbiguousExample:
    def __init__(self, text, source=None):
        self.text = text
        self.source = source

    def get_truncated(self, max_remove=2):
        truncated_ambigs = []
        example_sentences = [s.strip() for s in self.text.strip().split('.') if s.strip()]
        non_unk_indices = [i for i in range(len(example_sentences)) if "<UNK>" not in example_sentences[i]]
        for i in range(min(len(non_unk_indices), max_remove)):
            reduced_sentences = [s for idx, s in enumerate(example_sentences) if idx not in non_unk_indices[:i+1]]
            new_ambig = AmbiguousExample('. '.join(reduced_sentences))
            truncated_ambigs.append(new_ambig)

        return truncated_ambigs

    def __json__(self):
        return {
            "text": self.text,
            "src": self.source,
        }

with open("prompts/nsm_multi_turn_examples.json", "r") as f:
    explication_examples = json.load(f)

class ChatFormat(Enum):
    DEFAULT = auto()  # role: user/assistant, content: ...
    GEMINI = auto()   # Content(role=..., parts=[Part.from_text(...)])

def is_gemini(fmt: ChatFormat) -> bool:
    return fmt == ChatFormat.GEMINI
 
def build_prompt_from_parts(
    system_instruction: str | None,
    few_shot_examples: list[tuple[str, str]] = None,
    user_query: str = "",
    format: ChatFormat = ChatFormat.DEFAULT,
    system_supported: bool = True,
    multi_turn_supported: bool = True,
) -> tuple[list, object]:
    """
    Constructs a list of messages + optional generation config for chat models.
    """
    messages = []
    generate_content_config = None
    full_text = ""

    if system_instruction:
        if system_supported:
            if is_gemini(format):
                generate_content_config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    system_instruction=[types.Part.from_text(text=system_instruction)]
                )
            else:
                messages.append({"role": "system", "content": system_instruction})
        elif not system_supported and multi_turn_supported:
            if is_gemini(format):
                messages.extend([
                    types.Content(role="user", parts=[types.Part.from_text(text=f"<SYS>{system_instruction}<SYS>")]),
                    types.Content(role="model", parts=[types.Part.from_text(text="System Instruction Received.")])
                ])
            else:
                messages.extend([
                    {"role": "user", "content": f"<SYS>{system_instruction}<SYS>"},
                    {"role": "assistant", "content": "System Instruction Received."}
                ])
        else:
            full_text += f"<SYS>{system_instruction}<SYS>\n"

    if few_shot_examples:
        for user_msg, assistant_msg in few_shot_examples:
            if multi_turn_supported:
                if is_gemini(format):
                    messages.extend([
                        types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]),
                        types.Content(role="model", parts=[types.Part.from_text(text=assistant_msg)])
                    ])
                else:
                    messages.extend([
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ])
            else:
                full_text += f"<USER>{user_msg}<USER>\n<ASSISTANT>{assistant_msg}<ASSISTANT>\n"

    final_input = user_query if multi_turn_supported else f"{full_text}<USER>{user_query}<USER>"

    if is_gemini(format):
        messages.append(types.Content(role="user", parts=[types.Part.from_text(text=final_input)]))
    else:
        messages.append({"role": "user", "content": final_input})

    return messages, generate_content_config

def build_explication_prompt(word: str, sense_examples: list[str], format: ChatFormat,
                             system_supported=True, multi_turn_supported=True,
                             use_few_shot=True, max_few_shot: int = None):
    few_shot_examples = []

    if use_few_shot:
        max_count = min(len(explication_examples), max_few_shot or len(explication_examples))
        for i in range(max_count):
            user_msg = explication_examples[i][0]["content"]
            assistant_msg = explication_examples[i][1]["content"]
            few_shot_examples.append((user_msg, assistant_msg))

    user_query = EXPLICATION_USER_TEMPLATE.format(
        word=word,
        examples="\n\n".join(sense_examples)
    )

    return build_prompt_from_parts(
        system_instruction=NSM_EXPLICATION_SYS_INST,
        few_shot_examples=few_shot_examples,
        user_query=user_query,
        format=format,
        system_supported=system_supported,
        multi_turn_supported=multi_turn_supported
    )

# def build_example_prompt(entry: WordSenseEntry, num_examples=2, ambiguous=False, format:ChatFormat=ChatFormat.DEFAULT, system_supported=True, multi_turn_supported=True):
#     sys_msg = AMBIG_EXAMPLE_SYS_INST.format(num_examples=num_examples) if ambiguous else WORD_EXAMPLE_SYS_INST.format(num_examples=num_examples)

#     usr_msg = EXAMPLE_USER_TEMPLATE.format(word=entry.word, definition=entry.definition)

#     if entry.get_examples_from_sources("WordNet"):
#         usr_msg += " such as "
#         for i in range(len(entry.examples)):
#             usr_msg += f"'{entry.examples[i].text}'"
#             if i > 1:
#                 break
#             if i < len(entry.examples) - 1:
#                 usr_msg += " or "

#     return build_prompt_from_parts(
#         system_instruction=sys_msg,
#         few_shot_examples=None,
#         user_query=usr_msg,
#         format=format,
#         system_supported=system_supported,
#         multi_turn_supported=multi_turn_supported
#     )

def build_recover_prompt(ambig_example: AmbiguousExample, explication:Explication=None, format:ChatFormat=ChatFormat.DEFAULT, system_supported=True, multi_turn_supported=True):
    sys_msg = RECOVERY_PROMPT_EXPLICATION_SYS_INST if explication is not None else RECOVERY_PROMPT_SYS_INST
    usr_msg = RECOVERY_USER_TEMPLATE_EXPLICATION.format(passage=ambig_example.text, paraphrase=explication.text) if explication is not None else RECOVERY_USER_TEMPLATE.format(passage=ambig_example.text)

    return build_prompt_from_parts(
        system_instruction=sys_msg,
        few_shot_examples=None,
        user_query=usr_msg,
        format=format,
        system_supported=system_supported,
        multi_turn_supported=multi_turn_supported
    )

def gemini_prompt_to_jsonl(messages, generate_content_config, id=0):
    contents = []
    for msg in messages:
        if isinstance(msg, types.Content):
            parts = []
            for part in msg.parts:
                if hasattr(part, "text"):
                    parts.append({"text": part.text})
                else:
                    # Handle other part types if needed (e.g., images, functions)
                    pass
            contents.append({
                "role": msg.role,
                "parts": parts
            })
        else:
            raise ValueError(f"Unexpected message type: {type(msg)}")

    request_obj = {"id": id, "request": {"contents": contents}}

    # Include system_instruction as {"parts": [{"text": ...}]} at the same level as contents
    if generate_content_config and hasattr(generate_content_config, "system_instruction"):
        sys_parts = []
        for part in generate_content_config.system_instruction:
            if hasattr(part, "text"):
                sys_parts.append({"text": part.text})
        request_obj["request"]["system_instruction"] = {"parts": sys_parts}

    # Include generationConfig if available (excluding system_instruction)
    if generate_content_config:
        config_dict = {}
        if hasattr(generate_content_config, "response_mime_type"):
            config_dict["response_mime_type"] = generate_content_config.response_mime_type

        if config_dict:
            request_obj["request"]["generationConfig"] = config_dict

    return json.dumps(request_obj)

def openai_prompt_to_jsonl(messages, model_name, custom_id):
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": messages
        }
    })
