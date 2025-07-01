from huggingface_hub import login
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import logging

from dotenv import load_dotenv
from prompts import *
import torch
from accelerate import Accelerator as accelerator, PartialState


bfloat_supported = False
major, _ = torch.cuda.get_device_capability()
if major >= 8:
    PartialState().print("=" * 80)
    PartialState().print("Your GPU supports bfloat16, you can accelerate training with bf16")
    bfloat_supported = True
    PartialState().print("=" * 80)

load_dotenv()
login(os.getenv("HF_ACCESS_TOKEN"))

from peft import PeftModelForCausalLM

def print_banner():
    """Print a nice banner for the interface"""
    print("\n" + "="*80)
    print("🤖 DeepNSM Testing Interface")
    print("="*80)

def print_section_header(title):
    """Print a section header with nice formatting"""
    print(f"\n{'─'*60}")
    print(f"📋 {title}")
    print(f"{'─'*60}")

def print_success(message):
    """Print a success message with green color"""
    print(f"✅ {message}")

def print_info(message):
    """Print an info message with blue color"""
    print(f"ℹ️  {message}")

def print_warning(message):
    """Print a warning message with yellow color"""
    print(f"⚠️  {message}")

def get_model_selection():
    """Get user's model selection"""
    print_section_header("Model Selection")
    print("Available models:")
    print("  1. baartmar/DeepNSM-1B")
    print("  2. baartmar/DeepNSM-8B") 
    print("  3. meta-llama/Llama-3.1-8B-Instruct")
    print("  4. meta-llama/Llama-3.2-1B-Instruct")
    while True:
        try:
            choice = int(input("\nSelect a model (1-4): "))
            if 1 <= choice <= 4:
                models = [
                    "baartmar/DeepNSM-1B",
                    "baartmar/DeepNSM-8B",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct"
                ]
                selected_model = models[choice - 1]
                print_success(f"Selected: {selected_model}")
                return selected_model
            else:
                print_warning("Please enter a number between 1 and 4.")
        except ValueError:
            print_warning("Please enter a valid number.")

def get_word_and_examples():
    """Get word and examples from user"""
    print_section_header("Input Word & Examples")
    word = input("Enter a word to paraphrase using the NSM primes: ").strip()
    
    if not word:
        print_warning("Word cannot be empty.")
        return None, []
    
    print(f"\nEnter examples of '{word}' (type 'DONE' when finished):")
    examples = []
    
    while True:
        example = input(f"Example {len(examples) + 1}: ").strip()
        if example.upper() == 'DONE':
            break
        if example:
            examples.append(example)
        else:
            print_warning("Please enter a valid example or 'DONE'.")
    
    if not examples:
        print_warning("At least one example is required.")
        return None, []
    
    print_success(f"Added {len(examples)} example(s)")
    return word, examples

def generate_explication_prompt(word, examples):
    """Generate the explication prompt"""
    return f"""Word: {word}
Examples:
{"\n".join(examples)}
Paraphrase:
"""

def load_model_and_tokenizer(model_name):
    """Load and return the model and tokenizer for the given model_name."""
    global bfloat_supported
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModelForCausalLM

    # Suppress warnings during model loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        if model_name == "baartmar/DeepNSM-1B":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nsm_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16,
                device_map="auto"
            )
            nsm_model.resize_token_embeddings(len(tokenizer))
            nsm_model = PeftModelForCausalLM.from_pretrained(
                nsm_model,
                model_name
            )
            nsm_model = nsm_model.merge_and_unload()
            nsm_model.eval()
            return nsm_model, tokenizer

        elif model_name == "baartmar/DeepNSM-8B":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nsm_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16,
                device_map="auto"
            )
            nsm_model.resize_token_embeddings(len(tokenizer))
            nsm_model = PeftModelForCausalLM.from_pretrained(
                nsm_model,
                model_name
            )
            nsm_model = nsm_model.merge_and_unload()
            nsm_model.eval()
            return nsm_model, tokenizer

        elif "llama" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nsm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if bfloat_supported else torch.float16,
                device_map="auto"
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            nsm_model.eval()
            return nsm_model, tokenizer
        else:
            raise ValueError(f"Unknown model name: {model_name}")


def run_inference(nsm_model, tokenizer, model_name, word, examples):
    # Suppress warnings during inference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        if model_name in ["baartmar/DeepNSM-1B", "baartmar/DeepNSM-8B"]:
            prompt = f"""Word: {word}\nExamples:\n{"\n".join(examples)}\nParaphrase:"""
        elif "llama" in model_name.lower():
            prompt = build_explication_prompt(word, examples, ChatFormat.DEFAULT, max_few_shot=3)
            prompt = tokenizer.apply_chat_template(prompt[0], add_generation_prompt=True, tokenize=False)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = nsm_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id  # Explicitly set to suppress pad token warning
            )

        # Get only new tokens
        new_tokens = output_ids[0][input_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Calculate legality score
        explication = Explication(decoded)
        explication.legality_score(word)
        
        # Print results with nice formatting
        print_section_header("Generated Explication")
        print(f"📝 {decoded}")
        
        print_section_header("NSM Legality Analysis")
        
        # Create a nice table-like display
        print(f"{'Metric':<20} {'Value':<15} {'Details':<25}")
        print(f"{'─'*20} {'─'*15} {'─'*25}")
        
        # Primes ratio with color coding
        prime_color = "🟢" if explication.primes_ratio >= 0.6 else "🟡" if explication.primes_ratio >= 0.4 else "🔴"
        print(f"{'Primes Ratio':<20} {explication.primes_ratio:<15.3f} {prime_color} {explication.primes}/{explication.length} tokens")
        
        # Molecules ratio with color coding
        molecule_color = "🟢" if explication.molecules_ratio <= 0.2 else "🟡" if explication.molecules_ratio <= 0.4 else "🔴"
        print(f"{'Molecules Ratio':<20} {explication.molecules_ratio:<15.3f} {molecule_color} {explication.molecules}/{explication.length} tokens")
        
        # Stopwords
        stopword_color = "🟢" if explication.stop_words <= explication.length * 0.3 else "🟡" if explication.stop_words <= explication.length * 0.5 else "🔴"
        print(f"{'Stopwords':<20} {explication.stop_words:<15} {stopword_color} {explication.stop_words}/{explication.length} tokens")
        
        # Circular usage
        circular_color = "🔴" if explication.uses_original_word else "🟢"
        circular_status = "Yes (circular)" if explication.uses_original_word else "No (good)"
        print(f"{'Uses Original Word':<20} {circular_status:<15} {circular_color}")
        
        # Total length
        print(f"{'Total Length':<20} {explication.length:<15} tokens")
        
        # Summary
        print(f"\n{'─'*60}")
        if explication.uses_original_word:
            print_warning("⚠️  WARNING: Explication uses the original word (circular definition)")
        else:
            print_success("✅ Explication does not use the original word")
        
        if explication.primes_ratio >= 0.6:
            print_success("✅ High proportion of NSM primes")
        elif explication.primes_ratio >= 0.4:
            print_info("ℹ️  Moderate proportion of NSM primes")
        else:
            print_warning("⚠️  Low proportion of NSM primes")
        
        return decoded


def main():
    """Main user interaction loop"""
    print_banner()
    import torch
    nsm_model = None
    tokenizer = None
    model_name = None

    while True:
        # Get model selection
        new_model_name = get_model_selection()

        # If switching models, delete old model/tokenizer and clear cache
        if nsm_model is not None or tokenizer is not None:
            print_info("Cleaning up previous model...")
            del nsm_model
            del tokenizer
            torch.cuda.empty_cache()
            print_success("Memory cleared")

        # Load new model and tokenizer
        print_info("Loading model and tokenizer...")
        nsm_model, tokenizer = load_model_and_tokenizer(new_model_name)
        model_name = new_model_name
        print_success("Model loaded successfully!")

        while True:
            # Get word and examples
            word, examples = get_word_and_examples()
            if word is None:
                continue

            # Run inference
            print_info("Generating explication...")
            explication = run_inference(nsm_model, tokenizer, model_name, word, examples)
            
            # Ask user what to do next
            print_section_header("Next Steps")
            print("What would you like to do?")
            print("  1. Try another word with the same model")
            print("  2. Try a different model")
            print("  3. Exit")

            while True:
                try:
                    choice = int(input("\nEnter your choice (1-3): "))
                    if 1 <= choice <= 3:
                        break
                    else:
                        print_warning("Please enter a number between 1 and 3.")
                except ValueError:
                    print_warning("Please enter a valid number.")

            if choice == 1:
                print_info("Continuing with same model...")
                continue  # Try another word with same model
            elif choice == 2:
                print_info("Switching to different model...")
                break     # Try different model
            else:  # choice == 3
                print_section_header("Goodbye!")
                print("Thanks for using DeepNSM Testing Interface! 👋")
                # Clean up before exit
                del nsm_model
                del tokenizer
                torch.cuda.empty_cache()
                return

if __name__ == "__main__":
    main()