from datasets import Dataset
from huggingface_hub import login

access_token = ""
login(access_token)

data = [
    {
        "word": "happy",
        "examples": [
            "She felt happy when she saw her friends.",
            "The children were happy playing in the park."
        ],
        "ns_metalanguage": "Someone feels something good. This someone thinks: 'Something good happened. I wanted this. Because of this, I feel something good now.'"
    },
    {
        "word": "angry",
        "examples": [
            "He was angry because someone broke his toy.",
            "She gets angry when people are late."
        ],
        "ns_metalanguage": "Someone feels something bad. This someone thinks: 'Something bad happened. I don't want this. Because of this, I want to do something bad to someone.'"
    },
    {
        "word": "promise",
        "examples": [
            "I promise I will help you.",
            "She made a promise to her friend."
        ],
        "ns_metalanguage": "Someone says something like: 'I will do something because I want you to know this. If you want this, I will do it.'"
    }
]

dataset = Dataset.from_list(data)

repo_id = "raffelm/NSM_dataset"
dataset.push_to_hub(
repo_id,
commit_message="First NSM dataset",
private=True,
)

print(f"Dataset pushed to {repo_id} on Hugging Face Hub.") 