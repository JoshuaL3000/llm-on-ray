# LLM on Ray Workflow

## Introduction
There are many reasons that you may want to build and serve your own Large Language Models(LLMs) such as cost, latency, data security, etc. With more high quality open source models being released, it becomes possible to finetune a LLM to meet your specific needs. However, finetuning a LLM is not a simple task as it involves many different technologies such as PyTorch, Huggingface, Deepspeed and more. It becomes more complex when scaling to a cluster as you also need to take care of infrastructure management, fault tolerance, etc. Serving a small LLM model on a single node might be simple. But deploying a production level online inference service is also challenging.
In this workflow, we show how you can finetune your own LLM with your proprietary data and deploy an online inference service easily on an Intel CPU cluster.



## Solution Technical Overview
Ray is a leading solution for scaling AI workloads, allowing people to train and deploy models faster and more efficiently. Ray is used by leading AI organizations to train LLMs at scale (e.g., by OpenAI to train ChatGPT, Cohere to train their models, EleutherAI to train GPT-J). Ray provides the ability to recover from training failure and auto scale the cluster resource. Ray is developer friendly with the ability to debug remote tasks easily. In this workload, we run LLM finetuning and serving on Ray to leverage all the benefits provided by Ray. Meanwhile we also integrate LLM optimizations from Intel such as IPEX.


## Solution Technical Details
To finetune a LLM, usually you start with selecting an open source base model. In rare case, you can also train the base model yourself and we plan to provide a pretraining workflow as well in the future.  Next, you can prepare your proprietary data and training configurations and feed them to the finetuning workflow. After the training is completed, a finetuned model will be generated. The serving workflow can take the new model and deploy it on Ray as an online inference service. You will get a model endpoint that can be used to integrate with your own application such as a chatbot.

![image](https://github.com/intel-sandbox/llm-ray/assets/9278199/addd7a7f-83ef-43ae-b3ac-dd81cc2570e4)



## Hardware and Software requirements
### Hardware Requirements
### Software Requirements
- Docker 
- NFS 
- Python3
### Validated Hardware Details
There are workflow-specific hardware and software setup requirements depending on how the workflow is run. Bare metal development system and Docker image running locally have the same system requirements.

|Recommended Hardware|Precision|
|-|-|
|Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processo | FP32|

Workflow has been tested on Linux-4.18.0-408.el8.x86_64 and Ubuntu 22.04
## How it work

### Finetune
This finetune workflow can be configured by the user using configuration files and it supports running in different ways:

+ Run single node bare metal
+ Run ray cluster
The selection between these different modes can be done in the Finetune/llm_finetune_template.conf.

Update Finetune/llm_finetune_template.conf.
llm_finetune_template.conf is the main configuration file for the user to specify:

+ Runtime environment (i,e number of nodes in cluster, IPs, bare metal/docker, ...)
+ Directories for inputs, outputs and configuration files
Configure what stages of the workflow to execute. A user may run all stages the first time but may want to skip building or partitioning a graph in later training experiments to save time.
Please refer to the Finetune/llm_finetune_template.conf for a detailed description.

### Inference
todo


## Get Started
### Download the Workflow Repository
Create a working directory for the workflow and clone the Main Repository repository into your working directory.

mkdir ~/workspace && cd ~/workspace
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray
git checkout main

### Install dependencies

pip install -r requirements.txt
pip install -r requirements.intel.txt -f https://developer.intel.com/ipex-whl-stable-cpu

### Launch ray cluster
#### head node
ray start --head --node-ip-address $ray_node-ip-address --ray-debugger-external
#### worker node
ray start --address='$ray_node-ip-address:port' --ray-debugger-external

If deploying a ray cluster on multiple nodes, please download the workflow repository on each node. More information about ray cluster, please refer to https://www.ray.io/

### Run Workflow
#### Finetune 
Modify some configuration items, include `trainer.output` `checkpoint.root_path` `ray_config.init._node_ip_address`. The above configurations are related to the operating environment. So when the operating environment changes, it needs to be modified.
Once the prerequisits have been met and the llm_finetune_template.conf files have been updated to execute the workflow, use these commands to run the workflow:
python Finetune/main.py --config_path Finetune/llm_finetune_template.conf 

#### Inference
todo

## Expected Output
The successful execution of this stage will create the below contents under `output` and `checkpoint` directory.
```
output/
|-- config.json
|-- generation_config.json
|-- pytorch_model-00001-of-00003.bin
|-- pytorch_model-00002-of-00003.bin
|-- pytorch_model-00003-of-00003.bin
`-- pytorch_model.bin.index.json
checkpoint/
|-- test_0-of-2
|   `-- dict_checkpoint.pkl
`-- test_1-of-2
    `-- dict_checkpoint.pkl
TorchTrainer_2023-06-05_08-50-46/
|-- TorchTrainer_10273_00000_0_2023-06-05_08-50-47
|   |-- events.out.tfevents.1685955047.localhost
|   |-- params.json
|   |-- params.pkl
|   |-- rank_0
|   |-- rank_1
|   `-- result.json
|-- basic-variant-state-2023-06-05_08-50-46.json
|-- experiment_state-2023-06-05_08-50-46.json
|-- trainable.pkl
|-- trainer.pkl
`-- tuner.pkl
```
## Customize
### Adopt to your dataset
You can bring your own dataset to be used with this workflow. The dataprocesser is packaged into an independent module in Finetune/plugin/dataprocesser. So users can define data processing methods according to their own data format. If you want to process wikitext data in a different way, then you can follow the example of wikitextprocesser and rewrite a class. At the same time, set the new class name to llm_finetune_template.conf.

