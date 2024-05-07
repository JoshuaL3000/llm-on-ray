from wrapper.llmray_class import llmray
from wrapper.configs import finetune_config, ray_init_config
from inference.inference_config import all_models
import os
import ray
from finetune.finetune import get_accelerate_environment_variable

accelerate_env_vars = get_accelerate_environment_variable(
    finetune_config["Training"]["accelerate_mode"], config=None
)

ray_init_config['address'] = "ray://172.31.89.180:10001"
print("Start to init Ray connection")
context = ray.init(**ray_init_config)
print("Ray connected")
# initial_model_list = {k: all_models[k] for k in sorted(all_models.keys())}
head_node_ip = "172.31.89.180"
initial_model_list = {k: all_models[k] for k in sorted(all_models.keys())}

llm_ray = llmray(
    initial_model_list, 
    initial_model_list, 
    "/home/ubuntu/llmray_easydata/153/",
    finetune_config, 
    head_node_ip) 

llm_ray.deploy ("llama-2-7b-chat-hf", "llama-2-test-22", replica_num=1, cpus_per_worker_deploy=40, hf_token="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")