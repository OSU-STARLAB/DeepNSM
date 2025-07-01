import argparse
from huggingface_hub import login
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator as accelerator, PartialState
import json
from tqdm import tqdm
from openai import OpenAI
from google import genai
from peft import PeftModelForCausalLM, PeftModel
import numpy as np
from datetime import datetime
import random

from prompts import *
from utils import *

random.seed(554)
torch.manual_seed(554)


login(os.getenv("HF_ACCESS_TOKEN"))

# ************************* CLASSES ******************************

class ModelResult:
    def __init__(self, model_name:str):
        self.model_name = model_name
        self.num_examples = 0
        
        self.explications:list[Explication] = []
        
        self.avg_length = 0.0
        self.avg_primes = 0.0
        self.avg_stop_words = 0.0
        self.avg_molecules = 0.0
        
        self.avg_primes_ratio = 0.0
        self.avg_molecules_ratio = 0.0

        self.uses_original_word_ratio = 0.0
        
        self.avg_delta = 0.0
        self.avg_delta_min = 0.0
        self.avg_delta_ent = 0.0

        self.avg_score_exp = 0.0
        self.avg_total_score = 0.0

        self.avg_comet = []
        self.avg_bleu = []
        self.avg_emb = []

    def calculate_averages(self):
        self.avg_length = sum([expl.length for expl in self.explications]) / len(self.explications)
        self.avg_primes = sum([expl.primes for expl in self.explications]) / len(self.explications)
        self.avg_stop_words = sum([expl.stop_words for expl in self.explications]) / len(self.explications)
        
        self.avg_molecules = sum([expl.molecules for expl in self.explications]) / len(self.explications)
        self.avg_primes_ratio = sum([expl.primes_ratio for expl in self.explications]) / len(self.explications)
        self.avg_molecules_ratio = sum([expl.molecules_ratio for expl in self.explications]) / len(self.explications)
        self.uses_original_word_ratio = sum([1 for expl in self.explications if expl.uses_original_word]) / len(self.explications)
        self.avg_delta = sum([expl.avg_delta for expl in self.explications]) / len(self.explications)
        self.avg_delta_min = sum([expl.avg_delta_min for expl in self.explications]) / len(self.explications)
        self.avg_delta_ent = sum([expl.avg_delta_ent for expl in self.explications]) / len(self.explications)
        self.avg_score_exp = sum([expl.score_exp for expl in self.explications]) / len(self.explications)
        self.avg_total_score = sum([expl.total_score for expl in self.explications]) / len(self.explications)

    def __json__(self):
        return {
            "model": self.model_name,
            "num_examples": self.num_examples,
            "avg_total_score": self.avg_total_score,
            "avg_primes_ratio": self.avg_primes_ratio,
            "avg_molecules_ratio": self.avg_molecules_ratio,
            "avg_circularity": self.uses_original_word_ratio,
            "avg_score_exp": self.avg_score_exp,
            "avg_comet": self.avg_comet,
            "avg_bleu": self.avg_bleu,
            "avg_emb": self.avg_emb,
            "avg_delta": self.avg_delta,
            "avg_delta_min": self.avg_delta_min,
            "avg_delta_ent": self.avg_delta_ent,
            "avg_length": self.avg_length,
            "avg_primes": self.avg_primes,
            "avg_stop_words": self.avg_stop_words,
            "avg_molecules": self.avg_molecules,
            "explications": [expl.__json__() for expl in self.explications]
        }

class SubstitutabilityScore:
    def __init__(self, 
                 model_or_dict:str|dict
                 ):
        if isinstance(model_or_dict, dict):
            self.model = model_or_dict.get("model", model_or_dict)
            self.baselines = [Prediction(pred) for pred in model_or_dict.get("baselines", [])]
            self.exp_baselines = [Prediction(pred) for pred in model_or_dict.get("exp_baselines", [])]
            
            self.minimality = []
            minimality_scores = model_or_dict.get("minimality", [])
            for min_pair in minimality_scores:
                pair = []
                for score in min_pair:
                    pair.append(Prediction(score))
                self.minimality.append(pair)

            self.entailments = []
            entailments_scores = model_or_dict.get("entailments", [])
            for ent_pair in entailments_scores:
                pair = []
                for score in ent_pair:
                    pair.append(Prediction(score))
                self.entailments.append(pair)
                
            self.adj_score = model_or_dict.get("adj_score", 0.0)
            self.avg_delta_log = model_or_dict.get("avg_delta_log", 0.0)
            self.avg_min_delta_log = model_or_dict.get("avg_min_delta_log", 0.0)
            self.avg_ent_delta_log = model_or_dict.get("avg_ent_delta_log", 0.0)
            self.total_match = model_or_dict.get("total_match", 0)
        else:
            self.model = model_or_dict
            self.baselines = []
            self.exp_baselines = []
            self.minimality = []
            self.entailments = []
            self.adj_score = 0.0
            self.avg_delta_log = 0.0
            self.avg_min_delta_log = 0.0
            self.avg_ent_delta_log = 0.0
            self.total_match = 0

    def __str__(self):
        pass

    def __json__(self):
        return {
            "model": self.model,
            "adj_score": self.adj_score,
            "avg_delta_log": self.avg_delta_log,
            "avg_min_delta_log": self.avg_min_delta_log,
            "avg_ent_delta_log": self.avg_ent_delta_log,
            "total_match": self.total_match,
            "baselines": [baseline.__json__() for baseline in self.baselines],
            "exp_baselines": [baseline.__json__() for baseline in self.exp_baselines],
            "minimality": [
                [score.__json__() for score in pair] for pair in self.minimality
            ],
            "entailments": [
                [score.__json__() for score in pair] for pair in self.entailments
            ]
        }

class Prediction:
    def __init__(self,
                 prediction_or_dict:str|dict,
                 answer_logprob=0.0,
                 answer_ranks:list[int]=[],
                 is_match=False,
                 lines_removed=0
                 ):
        if isinstance(prediction_or_dict, dict):
            self.prediction=prediction_or_dict.get("prediction", "")
            self.answer_logprob = prediction_or_dict.get("answer_logprob", answer_logprob)
            self.answer_ranks = prediction_or_dict.get("answer_ranks", answer_ranks)
            self.is_match= prediction_or_dict.get("is_match", is_match)
            self.lines_removed = prediction_or_dict.get("lines_removed", lines_removed)
        else:
            self.prediction=prediction_or_dict
            self.answer_logprob = answer_logprob
            self.answer_ranks = answer_ranks
            self.is_match=is_match
            self.lines_removed = lines_removed

    def __str__(self):
        return f"Predicted Word: {self.prediction} | Answer Logprob: {self.answer_logprob} | Answer Rank(s): {self.answer_ranks} | Is Match: {"Yes" if self.is_match else "No"} | Lines Removed: {self.lines_removed}"

    def __json__(self):
        return {
            "prediction": self.prediction,
            "answer_logprob": self.answer_logprob,
            "answer_ranks": self.answer_ranks,
            "is_match": self.is_match,
            "lines_removed": self.lines_removed
        }

# ************************* CLASSES ******************************

def prepare_batch_prompts(model_name, ambigs:list[AmbiguousExample], explications:list[Explication]):
    num_ambigs = len(ambigs)
    num_explications = len(explications)

    baseline_preds = [None for _ in range(num_ambigs)]
    baseline_prompts = []
    preds =  [[[None, None, None, None, None] for _ in range(num_ambigs)] for _ in range(num_explications)]
    batch_prompts = []

    for i, example in enumerate(ambigs):
        if "<UNK>" not in example.text: # if no UNK, we can't do a fair evaluation for this example, so skip
            continue

        baseline_preds[i] = Prediction("")
        prompt, _ = build_recover_prompt(example, system_supported=False if "gemma" in model_name else True)
        baseline_prompts.append(prompt)

        truncated_ambigs = example.get_truncated()

        for j, explication in enumerate(explications):
            
            preds[j][i][0] = Prediction("")
            prompt, _ = build_recover_prompt(example, explication, system_supported=False if "gemma" in model_name.lower() else True)
            batch_prompts.append(prompt)

            truncated_explications = explication.get_truncated()
            
            for k in range(len(truncated_explications)):
                preds[j][i][1+k] = Prediction("", lines_removed=1+k)
                prompt, _ = build_recover_prompt(example, truncated_explications[k], system_supported=False if "gemma" in model_name.lower() else True)
                batch_prompts.append(prompt)

            for k in range(len(truncated_ambigs)):
                preds[j][i][3+k] = Prediction("", lines_removed=1+k)
                prompt, _ = build_recover_prompt(truncated_ambigs[k], explication, system_supported=False if "gemma" in model_name.lower() else True)
                batch_prompts.append(prompt)

    return baseline_preds, preds, baseline_prompts + batch_prompts

def make_predictions(model, tokenizer, word, all_prompts, baseline_preds, preds, batch_size):
    texts = tokenizer.apply_chat_template(all_prompts, add_generation_prompt=True, tokenize=False)

    # Tokenize the full batch of inputs once
    all_inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    input_ids_full = all_inputs["input_ids"]
    attention_mask_full = all_inputs["attention_mask"]
    prefix_len = input_ids_full.shape[1]

    # Precompute the two possible token sequences for the target word
    word_tok          = tokenizer.tokenize(word)
    word_tok_cap      = tokenizer.tokenize(word.capitalize())
    word_ids          = tokenizer.convert_tokens_to_ids(word_tok)
    word_ids_cap      = tokenizer.convert_tokens_to_ids(word_tok_cap)

    # Build a single flat task list, in the exact order your old code would have consumed them
    # Each entry describes where to store one prediction, and which "lines_removed" to pass
    tasks = []
    # First all the baseline slots
    for i in range(len(baseline_preds)):
        if baseline_preds[i] is not None:
            tasks.append(("baseline", i, None))
    # Then all the preds slots (with their lr tags)
    for i in range(len(baseline_preds)):
        for j in range(len(preds)):
            for k in range(5):
                if preds[j][i][k] is not None:
                    # same lr logic as before
                    lr = 1 if k in (1, 3) else 2 if k in (2, 4) else 0
                    tasks.append(("preds", j, i, k, lr))

    def _compute_prediction(seq_ids, score_list, lr=None):
        # seq_ids is a 1D tensor of input_ids + generated ids
        gen_ids = seq_ids[prefix_len:]
        toks    = tokenizer.convert_ids_to_tokens(gen_ids)
        toks_l   = [t.lower() for t in toks]

        # original vs capitalized log‐probs
        log_w, ranks_w = 0.0, []
        for idx, logits in enumerate(score_list[:len(word_ids)]):
            logp = torch.log_softmax(logits.float(), dim=-1)[word_ids[idx]].item()
            rank = (torch.argsort(logits, descending=True) == word_ids[idx]).nonzero(as_tuple=True)[0].item() + 1
            log_w += logp; ranks_w.append(rank)

        log_c, ranks_c = 0.0, []
        for idx, logits in enumerate(score_list[:len(word_ids_cap)]):
            logp = torch.log_softmax(logits.float(), dim=-1)[word_ids_cap[idx]].item()
            rank = (torch.argsort(logits, descending=True) == word_ids_cap[idx]).nonzero(as_tuple=True)[0].item() + 1
            log_c += logp; ranks_c.append(rank)

        # pick whichever is more likely
        if log_w >= log_c:
            match = toks_l == [t.lower() for t in word_tok]
            return Prediction(tokenizer.convert_tokens_to_string(toks).strip(),
                              log_w, ranks_w, match, lr)
        else:
            match = toks_l == [t.lower() for t in word_tok_cap]
            return Prediction(tokenizer.convert_tokens_to_string(toks).strip(),
                              log_c, ranks_c, match, lr)

    with torch.no_grad():
        # Helper to run one “mini‐batch” through generate() and immediately consume its items
        def _process_batch(in_ids, attn_mask):
            nonlocal tasks
            generated = model.generate(
                input_ids=in_ids,
                attention_mask=attn_mask,
                max_new_tokens=max(len(word_ids), len(word_ids_cap)),
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )
            seqs   = generated.sequences            # [B, prefix+gen]
            scores = generated.scores               # list of [B, vocab_size]
            B = seqs.size(0)
            # for each item in this batch, pop the next task and fill it
            for b in range(B):
                task = tasks.pop(0)
                if task[0] == "baseline":
                    _, idx, _ = task
                    baseline_preds[idx] = _compute_prediction(seqs[b], [s[b] for s in scores])
                else:
                    _, j, i, k, lr = task
                    preds[j][i][k] = _compute_prediction(seqs[b], [s[b] for s in scores], lr)

        # if small or no batching, do it all at once
        if batch_size is None or batch_size >= input_ids_full.size(0):
            _process_batch(input_ids_full, attention_mask_full)
        else:
            N = input_ids_full.size(0)
            for start in range(0, N, batch_size):
                end = start + batch_size
                _process_batch(input_ids_full[start:end],
                               attention_mask_full[start:end])

    del input_ids_full
    del attention_mask_full
    torch.cuda.empty_cache()
    return baseline_preds, preds

def compute_scores(model_name, baseline_preds, preds):
    explication_scores = []
    for j, pred_group in enumerate(preds):
        score = SubstitutabilityScore(model_name)
        score.baselines = baseline_preds
        deltas, deltas_min, deltas_ent = [], [], []
        total_match = 0

        for i, baseline in enumerate(baseline_preds):
            if baseline is None:
                continue
            
            exp_baseline, exp_min1, exp_min2, exp_ent1, exp_ent2 = pred_group[i]
            #print(pred_group[i])
            ents, mins = [], []

            score.exp_baselines.append(exp_baseline)
            if exp_baseline.is_match:
                total_match += 1

            deltas.append(exp_baseline.answer_logprob - baseline.answer_logprob)

            if preds[j][i][1] is not None:
                exp_min1 = preds[j][i][1]
                if exp_min1.is_match:
                    total_match += 1
                deltas_min.append(exp_min1.answer_logprob - exp_baseline.answer_logprob)
                mins.append(exp_min1)

            if preds[j][i][2] is not None:
                exp_min2 = preds[j][i][2]
                if exp_min2.is_match:
                    total_match += 1
                deltas_min.append(exp_min2.answer_logprob - exp_min1.answer_logprob)
                mins.append(exp_min2)
            
            score.minimality.append(mins)
            
            if preds[j][i][3] is not None:
                exp_ent1 = preds[j][i][3]
                if exp_ent1.is_match:
                    total_match += 1
                deltas_ent.append(exp_ent1.answer_logprob - exp_baseline.answer_logprob)
                ents.append(exp_ent1)

            if preds[j][i][4] is not None:
                exp_ent2 = preds[j][i][4]
                if exp_ent2.is_match:
                    total_match += 1
                deltas_ent.append(exp_ent2.answer_logprob - exp_ent1.answer_logprob)
                ents.append(exp_ent2)

            score.entailments.append(ents)

        score.avg_delta_log = sum(deltas) / len(deltas)
        score.avg_min_delta_log = sum(deltas_min) / len(deltas_min)
        score.avg_ent_delta_log = sum(deltas_ent) / len(deltas_ent)
        score.adj_score = min(40.0, score.avg_delta_log - score.avg_min_delta_log + score.avg_ent_delta_log)
        score.total_match = total_match
        explication_scores.append(score)
    
    return explication_scores

def cross_translate_eval(language_list, results):
    # Do cross-translate test for each lang for all predictions
    if language_list:
        from google.cloud import translate_v3
        from sacrebleu import sentence_bleu
        from sentence_transformers import SentenceTransformer, util
        from comet import download_model, load_from_checkpoint
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        comet_model.eval().cuda()
        translate_client = translate_v3.TranslationServiceClient()
        embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight + fast
        embedder.eval()
        embedder.cuda()
        PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

        def compute_comet(srcs: list[str], mt: list[str], refs: list[str]) -> tuple[list[float], float]:
            data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, mt, refs)]
            scores = comet_model.predict(data, batch_size=64, gpus=1)["scores"]
            return scores, float(np.mean(scores))
        
        def roundtrip(text: str, lang: str) -> str:
            try:
                response_a = translate_client.translate_text(
                    contents=[text],
                    parent=f"projects/{PROJECT_ID}/locations/global",
                    mime_type="text/plain",
                    source_language_code="en",
                    target_language_code=lang
                )
                forward = response_a.translations[0].translated_text
                response_back = translate_client.translate_text(
                    contents=[forward],
                    parent=f"projects/{PROJECT_ID}/locations/global",
                    mime_type="text/plain",
                    source_language_code=lang,
                    target_language_code="en"
                )
                back = response_back.translations[0].translated_text
                return back
            except Exception as e:
                print(f"Translation error: {e}")
                return ""
            
        def compute_bleu(orig: list[str], back: list[str]) -> tuple[list[float], float]:
            scores = [sentence_bleu(hyp, [ref]).score for hyp, ref in zip(back, orig)]
            return scores, float(np.mean(scores))

        def compute_embed_sim(orig: list[str], back: list[str]) -> tuple[list[float], float]:
            emb1 = embedder.encode(orig, convert_to_tensor=True)
            emb2 = embedder.encode(back, convert_to_tensor=True)
            sims = util.cos_sim(emb1, emb2)
            diag_scores = sims.diag().tolist()
            return diag_scores, float(np.mean(diag_scores))
        
        for lang in language_list:
            for model_pred in results:
                expls = [expl.text for expl in results[model_pred].explications]
                bt = [roundtrip(expls[i], lang) for i in range(len(expls))]
                bleu_scores, avg_bleu = compute_bleu(expls, bt)
                sim_scores, avg_sim = compute_embed_sim(expls, bt)
                comet_scores, avg_comet = compute_comet(expls, bt, expls)
                for i, expl in enumerate(results[model_pred].explications):
                    expl.ct_bleu_scores.append(bleu_scores[i])
                    expl.ct_comet_scores.append(comet_scores[i])
                    expl.ct_embed_scores.append(sim_scores[i])

                results[model_pred].avg_bleu.append(avg_bleu)
                results[model_pred].avg_comet.append(avg_comet)
                results[model_pred].avg_emb.append(avg_sim)
                print(f"\n[{lang}] {model_pred}")
                print(f"BLEU: {avg_bleu:.2f}, EmbedSim: {avg_sim:.2f}, COMET: {avg_comet:.2f}")

def substitutability_test(model_list, dataset, bfloat_supported):
    results = {}

    result = ModelResult("wordnet")
    for entry in tqdm(dataset, f"Baseline of WordNet Definitions"):
        result.explications.append(Explication(entry["def"]))
    
    results["wordnet_definitions"] = result

    for model_entry in model_list:
        model_name = model_entry["model"]
        use_system_prompt = model_entry["use_system_prompt"]
        num_examples = model_entry["num_examples"]
        orig_model_name = model_entry["orig"] if model_entry["orig"] else None
        max_batch_size = model_entry["max_batch_size"] if model_entry["max_batch_size"] else 0
        result = ModelResult(model_name)
        result.num_examples = num_examples

        if "gpt" in model_name.lower():
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            for entry in tqdm(dataset, f"Making predictions for {model_name}"):
                prompt, _ = build_explication_prompt(entry["word"], entry["examples"], ChatFormat.DEFAULT, system_supported=use_system_prompt, max_few_shot=num_examples)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    stream=False
                )
                text = response.choices[0].message.content
                result.explications.append(Explication(text))

        elif "gemini" in model_name.lower():
            client = genai.Client()
            for entry in tqdm(dataset, f"Making predictions for {model_name}"):
                messages, config = build_explication_prompt(entry["word"], entry["examples"], ChatFormat.GEMINI, system_supported=use_system_prompt, max_few_shot=num_examples)
                response = client.models.generate_content(
                    model=model_name,
                    contents=messages,
                    config=config
                )
                new_exp = Explication(response.text)
                result.explications.append(new_exp)

        elif any(["llama" in model_name.lower(), "mistral" in model_name.lower()]):
            all_prompts = []
            for entry in dataset:
                prompt, _ = build_explication_prompt(entry["word"], entry["examples"], ChatFormat.DEFAULT, system_supported=use_system_prompt, max_few_shot=num_examples)
                all_prompts.append(prompt)
                
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16, device_map="auto")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            all_prompts = tokenizer.apply_chat_template(all_prompts, add_generation_prompt=True, tokenize=False)

            # batching
            for i in tqdm(range(0, len(all_prompts), max_batch_size), f"Making predictions for {model_name}"):
                batch_prompts = all_prompts[i:i + max_batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                input_length = inputs.input_ids.shape[1]  # Length of the prompt

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7,
                        num_return_sequences=1
                    )

                # decode each generated sequence, only the new tokens
                for output, _ in zip(output_ids, inputs.input_ids):
                    # Get only new tokens
                    new_tokens = output[input_length:]
                    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    result.explications.append(Explication(decoded))

            del model
            del tokenizer
            torch.cuda.empty_cache()
            #process all prompts in batches max_batch_size

        elif "deepnsm" in model_name.lower():
            
            #set up batch prompt in batch size
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nsm_model = AutoModelForCausalLM.from_pretrained(orig_model_name, torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16, device_map="auto")
            nsm_model.resize_token_embeddings(len(tokenizer))
            nsm_model = PeftModelForCausalLM.from_pretrained(
                nsm_model,
                model_name
            )
            nsm_model = nsm_model.merge_and_unload()
            nsm_model.eval()

            all_prompts = []
            for entry in dataset:
                prompt = f"""Word: {entry["word"]}
                Examples:
                {"\n".join(entry["examples"])}
                Paraphrase:
                """
                all_prompts.append(prompt)

            # batching
            for i in tqdm(range(0, len(all_prompts), max_batch_size), f"Making predictions for {model_name}"):
                batch_prompts = all_prompts[i:i + max_batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                input_length = inputs.input_ids.shape[1]  # Length of the prompt

                with torch.no_grad():
                    output_ids = nsm_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7,
                        num_return_sequences=1
                    )

                # decode each generated sequence, only the new tokens
                for output, _ in zip(output_ids, inputs.input_ids):
                    # Get only new tokens
                    new_tokens = output[input_length:]
                    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    result.explications.append(Explication(decoded))
            
            del nsm_model
            del tokenizer
            torch.cuda.empty_cache()

        results[model_name] = result
    
    return results

def nsm_evaluation(eval_config):
    model_list = eval_config["models"]
    grader_list = eval_config["grader_models"]
    output_path = eval_config["output_path"]
    language_list = eval_config["languages"]
    
    bfloat_supported = False
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        PartialState().print("=" * 80)
        PartialState().print("Your GPU supports bfloat16, you can accelerate training with bf16")
        bfloat_supported = True
        PartialState().print("=" * 80)

    dataset = load_dataset(eval_config["dataset"]["name"], split=eval_config["dataset"]["split"])

    # do predictions for eval set for all models
    results = substitutability_test(model_list, dataset, bfloat_supported)

    # for model_entry in model_list:
    #     model_name = model_entry["model"]
    #     use_system_prompt = model_entry["use_system_prompt"]
    #     num_examples = model_entry["num_examples"]
    #     orig_model_name = model_entry["orig"] if model_entry["orig"] else None
    #     max_batch_size = model_entry["max_batch_size"] if model_entry["max_batch_size"] else 0
    #     result = ModelResult(model_name)

    #     if "gpt" in model_name.lower():
    #         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #         for entry in tqdm(dataset, f"Making predictions for {model_name}"):
    #             prompt, _ = build_explication_prompt(entry["word"], entry["examples"], ChatFormat.DEFAULT, system_supported=use_system_prompt, max_few_shot=num_examples)
    #             response = client.chat.completions.create(
    #                 model=model_name,
    #                 messages=prompt,
    #                 stream=False
    #             )
    #             text = response.choices[0].message.content
    #             result.explications.append(Explication(text))

    #     elif "gemini" in model_name.lower():
    #         client = genai.Client()
    #         for entry in tqdm(dataset, f"Making predictions for {model_name}"):
    #             messages, config = build_explication_prompt(entry["word"], entry["examples"], ChatFormat.GEMINI, system_supported=use_system_prompt, max_few_shot=num_examples)
    #             response = client.models.generate_content(
    #                 model=model_name,
    #                 contents=messages,
    #                 config=config
    #             )
    #             new_exp = Explication(response.text)
    #             result.explications.append(new_exp)

    #     elif any(["llama" in model_name.lower(), "mistral" in model_name.lower()]):
    #         all_prompts = []
    #         for entry in dataset:
    #             prompt, _ = build_explication_prompt(entry["word"], entry["examples"], ChatFormat.DEFAULT, system_supported=use_system_prompt, max_few_shot=num_examples)
    #             all_prompts.append(prompt)
                
    #         tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    #         model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16, device_map="auto")
    #         tokenizer.pad_token_id = tokenizer.eos_token_id
    #         all_prompts = tokenizer.apply_chat_template(all_prompts, add_generation_prompt=True, tokenize=False)

    #         # batching
    #         for i in tqdm(range(0, len(all_prompts), max_batch_size), f"Making predictions for {model_name}"):
    #             batch_prompts = all_prompts[i:i + max_batch_size]
    #             inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    #             input_length = inputs.input_ids.shape[1]  # Length of the prompt

    #             with torch.no_grad():
    #                 output_ids = model.generate(
    #                     **inputs,
    #                     max_new_tokens=512,
    #                     do_sample=True,
    #                     top_k=50,
    #                     top_p=0.95,
    #                     temperature=0.7,
    #                     num_return_sequences=1
    #                 )

    #             # decode each generated sequence, only the new tokens
    #             for output, _ in zip(output_ids, inputs.input_ids):
    #                 # Get only new tokens
    #                 new_tokens = output[input_length:]
    #                 decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #                 result.explications.append(Explication(decoded))

    #         #process all prompts in batches max_batch_size

    #     elif "nsllm" in model_name.lower():
    #         #set up batch prompt in batch size
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)
    #         nsm_model = LlamaForCausalLM.from_pretrained(orig_model_name).to("cuda")
    #         nsm_model.resize_token_embeddings(len(tokenizer))
    #         nsm_model = PeftModelForCausalLM.from_pretrained(
    #             nsm_model,
    #             model_name
    #         )
    #         nsm_model = nsm_model.merge_and_unload()
    #         nsm_model.eval()

    #         all_prompts = []
    #         for entry in dataset:
    #             prompt = f"""Word: {entry["word"]}
    #             Examples:
    #             {"\n".join(entry["examples"])}
    #             Paraphrase:
    #             """
    #             all_prompts.append(prompt)

    #         # batching
    #         for i in tqdm(range(0, len(all_prompts), max_batch_size), f"Making predictions for {model_name}"):
    #             batch_prompts = all_prompts[i:i + max_batch_size]
    #             inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    #             input_length = inputs.input_ids.shape[1]  # Length of the prompt

    #             with torch.no_grad():
    #                 output_ids = nsm_model.generate(
    #                     **inputs,
    #                     max_new_tokens=512,
    #                     do_sample=True,
    #                     top_k=50,
    #                     top_p=0.95,
    #                     temperature=0.7,
    #                     num_return_sequences=1
    #                 )

    #             # decode each generated sequence, only the new tokens
    #             for output, _ in zip(output_ids, inputs.input_ids):
    #                 # Get only new tokens
    #                 new_tokens = output[input_length:]
    #                 decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #                 result.explications.append(Explication(decoded))

    #     results[model_name] = result

    for i, entry in tqdm(enumerate(dataset), f"Legality Scoring Explications"):
    # Do grading for each grader model for all predictions
        for predictor_model_result in results:
            expl = results[predictor_model_result].explications[i]
            expl.legality_score(entry["word"])
        
    for grader_model in grader_list:
        grader_model_name = grader_model["model"]
        max_batch_size = grader_model["max_batch_size"]
        
        tokenizer = AutoTokenizer.from_pretrained(grader_model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(grader_model_name, torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16, device_map="auto")
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # do baseline scores for each ambig 

        for i, entry in tqdm(enumerate(dataset), f"{grader_model_name} grading explications"):
            ambig_examples = [AmbiguousExample(ex) for ex in entry["ambig_examples"]]
            explications = [results[pred].explications[i] for pred in results]
            if not ambig_examples:
                continue
            baseline_preds, preds, all_prompts = prepare_batch_prompts(grader_model_name, ambig_examples, explications)
            baseline_preds, preds = make_predictions(model, tokenizer, entry["word"], all_prompts, baseline_preds, preds, max_batch_size)
            scores = compute_scores(grader_model_name, baseline_preds, preds)
            for j, predictor_model_result in enumerate(results):
                expl = results[predictor_model_result].explications[i]
                expl.sub_scores.append(scores[j])

        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Do cross-translate test for each lang for all predictions
    if language_list:
        cross_translate_eval(language_list, results)


    for i in tqdm(range(len(dataset)), f"Calculating Avg Scores"):
        for predictor_model_result in results:
            expl = results[predictor_model_result].explications[i]
            expl.calculate_averages()

    for predictor_model_result in results:
        result = results[predictor_model_result]
        result.calculate_averages()
        print(f"{predictor_model_result}: Avg Primes Ratio: {result.avg_primes_ratio} | Avg Molecules Ratio: {result.avg_molecules_ratio} | Uses Orig Word Ratio: {result.uses_original_word_ratio} | Avg Score {result.avg_total_score} | Avg BLEU: {result.avg_bleu} | Avg COMET: {result.avg_comet} | Avg Emb: {result.avg_emb}")



    def custom_encoder(obj):
        if hasattr(obj, "__json__"):
            return obj.__json__()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file_name = f"/eval_results_{timestamp}.json"
    results["timestamp"] = f"{timestamp}"
    results["graders"] = [grader["model"] for grader in grader_list]
    results["languages"] = language_list
    with open(output_path + result_file_name, "w", newline='') as output_file:
        json.dump(results, output_file, default=custom_encoder)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSM Evaluation...')
    parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()

    with open(args.config_path, "r", newline='') as jsonfile:
        eval_config = json.load(jsonfile)

    nsm_evaluation(eval_config)
