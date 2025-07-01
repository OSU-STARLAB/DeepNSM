import json
import numpy as np

eval_1_path = "../results/eval_results_2025-04-26_10-44-18.json"
eval_2_main = "../results/eval_results_2025-04-26_12-49-17.json"
eval_2_ab = "../results/eval_results_2025-05-13_16-11-38.json"

# Models
wordnet_definitions = "wordnet_definitions"
deepnsm_1b = "nsllm-1B-35/checkpoint-10000"
deepnsm_1b_random = "/nfs/hpc/share/baartmar/nsllm-1B-random/checkpoint-10000"
deepnsm_8b = "/nfs/hpc/share/baartmar/nsllm-8B-35/checkpoint-10000"
deepnsm_8b_random = "/nfs/hpc/share/baartmar/nsllm-8B-random/checkpoint-10000"
l8 = "meta-llama/Llama-3.1-8B-Instruct"
l1 = "meta-llama/Llama-3.2-1B-Instruct"
gemini = "gemini-2.0-flash"
gpt = "gpt-4o"

# Load JSON files
with open(eval_1_path, "r") as f:
    eval_1 = json.load(f)
with open(eval_2_main, "r") as f:
    eval_2_main = json.load(f)
with open(eval_2_ab, "r") as f:
    eval_2_ab = json.load(f)

# Language indices
lang_indices = [0, 1, 2, 3, 7]

def compute_and_print_std(model_name, scores):
    total_scores = [e["total_score"] for e in scores]
    
    # Substitutability, legality, primes ratio, molecules ratio
    substitutability_scores = [e["score_exp"] for e in scores]
    legality_scores = [10 * e["primes_ratio"] - e["molecules_ratio"] for e in scores]
    primes_ratio_scores = [e["primes_ratio"] for e in scores]
    molecules_ratio_scores = [e["molecules_ratio"] for e in scores]

    bleu_scores = {i: [] for i in lang_indices}
    emb_scores = {i: [] for i in lang_indices}
    
    for e in scores:
        for idx in lang_indices:
            bleu_scores[idx].append(e["bleu_scores"][idx])
            emb_scores[idx].append(e["embed_scores"][idx])

    print(f"\nModel: {model_name}")
    
    # Calculate the standard deviation and the error bars
    total_score_std = np.std(total_scores)
    substitutability_std = np.std(substitutability_scores)
    legality_std = np.std(legality_scores)
    primes_ratio_std = np.std(primes_ratio_scores)
    molecules_ratio_std = np.std(molecules_ratio_scores)
    
    print(f"Total Score Std: {total_score_std:.4f}, Error Bar: {total_score_std / 149:.4f}")
    print(f"Substitutability Score Std: {substitutability_std:.4f}, Error Bar: {substitutability_std / 149:.4f}")
    print(f"Legality Score Std: {legality_std:.4f}, Error Bar: {legality_std / 149:.4f}")
    print(f"Primes Ratio Score Std: {primes_ratio_std:.4f}, Error Bar: {primes_ratio_std / 149:.4f}")
    print(f"Molecules Ratio Score Std: {molecules_ratio_std:.4f}, Error Bar: {molecules_ratio_std / 149:.4f}")
    
    for idx in lang_indices:
        bleu_std = np.std(bleu_scores[idx])
        print(f"BLEU {idx} Std: {bleu_std:.4f}, Error Bar: {bleu_std / 149:.4f}")
    for idx in lang_indices:
        emb_std = np.std(emb_scores[idx])
        print(f"Embed Sim {idx} Std: {emb_std:.4f}, Error Bar: {emb_std / 149:.4f}")

# Eval 1 models
for model in [wordnet_definitions, deepnsm_1b, deepnsm_1b_random, deepnsm_8b, deepnsm_8b_random, l1, l8]:
    result = eval_1[model]
    compute_and_print_std(model, result["explications"])

# Eval 2 models (merge main and AB)
for model in [gpt, gemini]:
    result_main = eval_2_main[model]["explications"]
    result_ab = eval_2_ab[model]["explications"]
    merged = []

    for i in range(len(result_main)):
        merged_entry = {
            "total_score": result_main[i]["total_score"],
            "bleu_scores": result_main[i]["bleu_scores"][:4] + [0,0,0] + [result_ab[i]["bleu_scores"][0]],
            "embed_scores": result_main[i]["embed_scores"][:4] + [0,0,0] + [result_ab[i]["embed_scores"][0]],
            "score_exp": result_main[i]["score_exp"],
            "primes_ratio": result_main[i]["primes_ratio"],
            "molecules_ratio": result_main[i]["molecules_ratio"]
        }
        merged.append(merged_entry)

    compute_and_print_std(model, merged)