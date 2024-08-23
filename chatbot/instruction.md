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
# run container, '--nv' enable experimental Nvidia.support
singularity run --nv  ollama.sif
# get interactive shell within the container.
singularity shell ollama.sif
# test if ollama is running
curl http://localhost:11434
```
## 3. Run the corrsponding python script