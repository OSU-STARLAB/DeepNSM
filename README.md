# DeepNSM
DeepNSM is a large language model that has been fine-tuned for generating [Natural Semantic Metalanguage](https://en.wikipedia.org/wiki/Natural_semantic_metalanguage) explications of word-meanings. This repository includes the code, models, and experiment code for the paper "Towards Universal Semantics With Large Language Models."

The models used in the paper are uploaded on HuggingFace:

https://huggingface.co/baartmar/DeepNSM-1B
https://huggingface.co/baartmar/DeepNSM-8B
https://huggingface.co/baartmar/nsm_dataset

For more details, please see our preprint on arxiv:
```
@misc{baartmans2025universalsemanticslargelanguage,
      title={Towards Universal Semantics With Large Language Models}, 
      author={Raymond Baartmans and Matthew Raffel and Rahul Vikram and Aiden Deringer and Lizhong Chen},
      year={2025},
      eprint={2505.11764},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.11764}, 
}
```

# Requirements
1. Python 3.12.5. The code for this repository is tested on Python 3.12.5. If you would like to use newer versions of python, you may need to relax some of the version constraints on `requirements.txt` to do so.
2. NVIDIA GPU with CUDA Support. A card with > 16GB VRAM is likely required to run full experiments.
3. You will probably need at least 8GB of free disk space to install the packages and download model weights, but much more is likely needed for running full experiments.


# First-Time Setup
1. Create a virtual environment (recommended) and install dependencies
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Create a .env file and fill it in with the correct values and API keys. To run the 
```
cp .env.example .env
```
3. (Optional) If you run a 50 series GPU or newer, you will likely need to install Torch 2.7 over the version provided in requirements.txt in order to run the LLMs properly.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

# Start-Up
Activate the virtual environment you created and the env file
```
source .venv/bin/activate
source .env
```

# Try out DeepNSM.
Colab Demo: https://colab.research.google.com/drive/1kWesMSQOgKOsXxONvZyinpdgh86gDBcy?usp=drive_link

To run the DeepNSM models on your machine, you will need to have followed the setup guide for this repository. You will also need a NVIDIA GPU capable of running inference on up to 8B parameter LLMs, if you would like to try the 8B variants. This script also allows you to try out DeepNSM-1B and Llama-3.2-1B for generating NSM explications.
```
python test_deepnsm.py
```

# Run Experimental Evaluation
Follow setup and startup instructions, then run the following script.
```
mkdir results
python nsm_evaluation.py --config_path eval_config.json
```
You can view the config JSON to see what models are being used for testing and evaluation. This will take some time to run and will require a moderately strong GPU, if you are running DeepNSM or Llama models locally. The results will be stored in a folder called "results."

# Llama3 Fine-Tuning for NSM
## Installation
To run install the dependencies with `pip install -r requirements.txt`.
## Example
Run the following script for fine-tuning. 
```
python3 train.py
	--model meta-llama/Llama-3.2-1B --training-set baartmar/nsm_dataset
	--lora-alpha 16 --lora-dropout 0.1 --lora-r 64 --peft
	--use-4bit --bnb-4bit-compute-dtype bfloat16 --bnb-4bit-quant-typenf4 --bnb
	--bsz 64 --update-freq 1 --optim paged_adamw_32bit --lr 2e-4 --lr-scheduler inverse_sqrt
	--warmup-ratio 0.03 --max-grad-norm 0.3 
	--save-interval 1000 --eval-interval 1000 --log-interval 1000
	--max-seq-length 256 --save-strategysteps --num-train-epochs 1
	--output-dir ${SAVE_DIR} 
	--eval-strategy steps --train 
