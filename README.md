# First-Time Setup
1. Create a virtual environment (recommended) and install dependencies
```
python -m venv .venv
pip install -r requirements.txt
```
2. Create a .env file and fill it in with the correct API keys.
```
cp -r .env.example .env
```

# Start-Up
Activate the virtual environment you created and the env file
```
source .venv/bin/activate
source .env
```

# Try out DeepNSM. To run the DeepNSM models on your machine, you will need to have followed the setup guide for this repository. You will also need a NVIDIA GPU capable of running inference on up to 8B parameter LLMs.
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