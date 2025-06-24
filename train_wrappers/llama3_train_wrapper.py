import torch
from transformers import AutoTokenizer

from transformers import LlamaForCausalLM
from peft import LoraConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from functools import partial

from argparse import ArgumentParser, Namespace

from train_wrappers.train_wrapper import LLMSFTTrainerWrapper

'''
The below class serves as a wrapper for fine-tuning Llama for simultaneous translation
via SFTTrainer. This extends from LLMSimulSFTTrainerWrapper and implements remaining 
unimplemented behavior from the parent wrapper.
'''

class LlamaSFTTrainerWrapper(LLMSFTTrainerWrapper):
    def __init__(self, args: Namespace):
        super().__init__(args)

    
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)

    def setup_peft_config(self, args):
        self.peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ) 


    def setup_model_and_tokenizer(self, args):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.bnb else None,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remove_code=True,
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = 'right'
 

    def setup_trainer(self, args):
        self.load_dataset()

        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        
        formatting = partial(formatting_func) 

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.training,
            eval_dataset=self.validation,
            peft_config=self.peft_config if self.peft else None,
            processing_class=self.tokenizer,
            data_collator=collator,
            formatting_func=formatting,
            args=self.training_arguments,
        )
   

    def train(self):
        self.trainer.train()
   

'''

Formatting function takes care of prompt specification for a given LLM and allows the data
collator to handle our data better. Example sentence at start of wait-3 translation:

   <h>: Given the English sentence {I'll tell you}, and the current translation in Spanish {},
   what's the next translated word? <a>: {Les}

'''

def formatting_func(example):
    output_texts = []  

    if not isinstance(example['word'], list) or not isinstance(example['examples'][0], list) or not isinstance(example['explication'], list):
        example['word'] = [example['word']]
        example['examples'] = [example['examples']]
        example['explication'] = [example['explication']]
    
    for word, examples, explication in zip(example['word'], example['examples'], example['explication']):
        examples = "\n".join(examples)
        text = f"<|start_header_id|>user<|end_header_id|>\nWord:{word}\nExamples:{examples}\nParaphrase:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{explication}"
        output_texts.append(text)

    return output_texts 


