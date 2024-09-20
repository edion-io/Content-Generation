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

# Convert model to GGUF format and quantizing the GGUF model
## 1. Papare enviroment
```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

## 2. Download model
```shell
# huggingface-cli login (if not login yet)
git clone https://huggingface.co/kangsive/llama3.1-8b-chat-exercise-tool_call
```

## 3. Convert to GGUF
```shell
python convert-hf-to-gguf.py llama3.1-8b-chat-exercise-tool_call \
    --outfile chat_exercise.gguf \
    -- outtype f16
```

## 4. Quantization
```shell
# Login to GPU node (make sure have access to GPU); this is important, otherwise quantization will be killed during runing
srun -N 1 --ntasks-per-node=1 --time=02:00:00 -p gpu --gpus-per-node=1 --nodes=1 --pty bash -i
# All available precision check in https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987
./llama-quantize chat_exercise.gguf chat_exercise_Q4_K_M.gguf Q4_K_M
```

## 5. Push to huggingface
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="/home1/s4702328/llama.cpp/chat_exercise_Q4_K_M.gguf",
    path_in_repo="llama3.1-8b-chat-exercise_tool_call-Q4_K_M.gguf",
    repo_id="kangsive/llama3.1-8b-chat-exercise-tool_call-gguf",
    repo_type="model",
)
```