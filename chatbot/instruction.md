# Chatbot Run Instruction
## 1. Install necessary python libraries
```shell
pip install langchain
pip install -u langchain_ollama
```
## 2. Install and run ollama:
```shell
# pull docker image and convert to singularity container.
singularity build ollama.sif docker://ollama/ollama
# run container, '--nv' enable experimental Nvidia.support.
singularity run --nv  ollama.sif
# get interactive shell within the container.
singularity shell ollama.sif
# test if ollama is running.
curl http://localhost:11434
```
## 3. Run the corresponding python script

# Creating synthetic dataset and fine-tune
## 1. Install necessary python libraries
```shell
pip install -u transformers
pip install -u datasets
pip install -u accelerate
pip install -u peft 
pip install -u trl 
pip install -u bitsandbytes
```
## 2. Create synthetic dataset
Run `create_synthetic_dataset.ipynb` or turn it into python scrpit before running
Run `apply_template.py` to apply llama3.1 template to conversations and save generated text as new data (Optinal for tool calling) 

## 3. Fine tuning
Run `huggingface-cli login` in terminal to login hugging face (source of target models) with your token, create one if you don't have it in `https://huggingface.co/settings/tokens`, then run `fine_fune.py`. If you don't have a local dataset, use `dataset = load_dataset("kangsive/exercise-generation-chat")` where you'll pull a corresponding dataset from huggingface if you've logined the account `kangsive`

## 4. Reload and test
Run `load_test.py` or Run `tool_call_test.py` (Optinal for tool calling)