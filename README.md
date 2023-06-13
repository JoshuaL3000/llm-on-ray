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

There are workflow-specific hardware and software setup requirements depending on how the workflow is run.
### Hardware Requirements

|Recommended Hardware|Precision|
|-|-|
|Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processo | FP32|

### Software Requirements
Workflow has been tested on Linux-4.18.0-408.el8.x86_64 and Ubuntu 22.04
- Docker
- NFS 
- Python3 > 3.7.16

## Run This Workflow

### Finetune
#### 1. Download the Workflow Repository
Create a working directory for the workflow and clone the repository into your working directory.
```bash
mkdir ~/workspace && cd ~/workspace
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray
git checkout main
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements.intel.txt -f https://developer.intel.com/ipex-whl-stable-cpu
```

#### 3. Launch ray cluster
#### head node
```bash
ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external
```
#### worker node
```bash
ray start --address='127.0.0.1:6379' --ray-debugger-external
```

If deploying a ray cluster on multiple nodes, please download the workflow repository on each node. More information about ray cluster, please refer to https://www.ray.io/

#### 4. Prepare Dataset

Now, the workflow supports two types of datasets. 


The first is plain text data similar to [wikitext](https://huggingface.co/datasets/wikitext). This type of data is used for finetuning in non-prompt mode and this type of data is characterized by containing `text` field. All the text under the `text` field will be directly used as finetuning data. Since most of the samples in these dataset are of different lengths, we provide switch named `group` to control whether to splice the data into the same length. 


The second is instruction fintuning dataset similar to [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k). This type of data is used for finetuning in prompt mode and this type of data is characterized by containing `instruction` `context` and `response` fields where `instruction` and `response` are required fields and `context` is an optional field. In the data preprocessing stage, the three fields will be concatenated to the corresponding format according to [dolly](https://github.com/databrickslabs/dolly/blob/master/training/trainer.py#LL93).


The meaning of the above three columns:
+ Instruction Column: The column in the dataset is the user input, such as a question or a command.
+ Context Column: This column is other information used by instruction, such as the options used in the question and so on. It can be empty.
+ Response: The column in the dataset containing the expected output.


Therefore, if the your data meets the above two formats, you can use the data by configuring the local data path or huggingface dataset. If not, please refer to the following **Adopt to Your Dataset**.

#### 5. Finetune

The workflow is designed for configure driven. All argument about the workflow are record to just one configure. So user can launch a task with a simple common command. Once the prerequisits have been met, use the following commands to run the workflow:
```bash
python Finetune/main.py --config_path Finetune/example/llm_finetune_template.conf 
```

Except `Finetune/llm_finetune_template.conf`, the repo also provide `Finetune/example/dolly_1_finetune.conf` and `Finetune/example/dolly_2_finetune.conf` for reproducing `dolly` rapidly. User can launch them with the same way.
```bash
python Finetune/main.py --config_path Finetune/example/dolly_1_finetune.conf
python Finetune/main.py --config_path Finetune/example/dolly_2_finetune.conf
```

If you want to finetune other model or change other dataset, you can build you own configure according to `Finetune/example/llm_finetune_template.conf `. 

#### 6. Expected Output
The successful execution of this stage will create the below contents under `output` and `checkpoint` directory. Model for deploying is in `output`. `checkpoint` help us to recovery from fault. `TorchTrainer_xxx` is generated by ray cluster automatically.
```
/tmp/output/
|-- config.json
|-- generation_config.json
|-- pytorch_model-00001-of-00003.bin
|-- pytorch_model-00002-of-00003.bin
|-- pytorch_model-00003-of-00003.bin
`-- pytorch_model.bin.index.json
/tmp/checkpoint/
|-- test_0-of-2
|   `-- dict_checkpoint.pkl
`-- test_1-of-2
    `-- dict_checkpoint.pkl
./TorchTrainer_2023-06-05_08-50-46/
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


### Inference
The inference workflow provides two execution methods, deploying it by UI or terminal execution.
#### Deploy by UI
![image](https://github.com/intel-sandbox/llm-ray/assets/97155466/f4ce763b-9f95-4f15-ae69-a69c514e9c88)

This method can launch a UI interface and deploy an online inference service with just a few simple clicks.
- Update `Inference/conf_file/llm_finetune_template.conf` as described in [Finetune](#Finetune).
- If a custom model needs to be added, please update `Inference/config.py`.
#### Terminal Execution
Ray serve is used to deploy model. First expose the model over HTTP using Deployment, then test it over HTTP request.

Update `Inference/config.py` as needed. Or model can be deployed by specifying the model path and tokenizer path.

#### Inference
**Deploy by UI**
```bash
python start_ui.py

# Running on local URL:  http://0.0.0.0:8080
# Running on public URL: https://180cd5f7c31a1cfd3c.gradio.live
```
Access url and deploy service in a few simple clicks.

**Terminal Execution**

You can deploy a custom model by passing parameters.
```bash
python run_model_serve.py --model $model --tokenizer $tokenizer

# INFO - Deployment 'custom-model_PredictDeployment' is ready at `http://127.0.0.1:8000/custom-model`. component=serve deployment=custom-model_PredictDeployment
# Service is deployed successfully

python run_model_infer.py --model_endpoint http://127.0.0.1:8000/custom-model
```
Or you can deploy models configured in `Inference/config.py` without passing parameters.


## Customize
### Adopt to your dataset
If the your data do not meets the above two supported formats, you may need to preprocess the data into the standard format. Here we provide an example dataset (converted dataset from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)) as `Finetune/process_data.py`. After running `python process_data.py`, a directory named `data` will output to the `Finetune` directory, just modify the configuration item `dataset.name` to the absolute path of this directory to start the task.

