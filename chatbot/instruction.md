# Chatbot Run Instruction
## 1. Load CUDA toolkit and install llama-cpp-python
```shell
# Load cuda toolkit (or any other way to make sure `nvcc --version` is working)
module load CUDA/12.1.1
# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install -U llama-cpp-python --force-reinstall --no-cache-dir
```
## 2. Run GGUF model iwth llama-cpp
```shell
module load CUDA/12.1.1
python echatbot.py 
```
## 3. Run API
```shell
# SERVE
module load CUDA/12.1.1
python api.py # or uvicorn api:app –reload

# REQUEST TEST
# Open another terminal with cuda toolkit loaded
curl -X GET http://127.0.0.1:8000/add_session   # {"session_id": abc}
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=Hello"
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=Math"
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=Multiple_choice"
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=5th_grade"
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=Nope"
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=Yes"
curl -X POST "http://127.0.0.1:8000/chat?session_id=abc&user_input=can_I_have_two_more_similar"
```

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
pip install -r requirements.txt
make -j8
```

## 2. Download model
```shell
# huggingface-cli login (if not login yet), and login to GPU node
git lfs install
git clone https://huggingface.co/kangsive/llama3.1-8b-chat-exercise-tool_call
cd llama3.1-8b-chat-exercise-tool_call
rm -rf .git
```

## 3. Convert to GGUF
```shell
python convert-hf-to-gguf.py llama3.1-8b-chat-exercise-tool_call \
    --outfile chat_exercise.gguf \
    -- outtype f16
# Exception: data did not match any variant of untagged enum ModelWrapper at line 1251003 column 3 --> Reinstall python envrionment.
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