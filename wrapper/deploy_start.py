import random, string, os, sys, argparse, socket
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ray
from ray import serve
from pydantic_yaml import parse_yaml_raw_as

from typing import Dict

import yaml
from configs import finetune_config, ray_init_config
from finetune.finetune import get_accelerate_environment_variable
from inference.inference_config import all_models, ModelDescription, Prompt, InferenceConfig
from inference.inference_config import InferenceConfig as FinetunedConfig
from inference.predictor_deployment import PredictorDeployment

accelerate_env_vars = get_accelerate_environment_variable(
    finetune_config["Training"]["accelerate_mode"], config=None
)

ray_init_config["runtime_env"]["env_vars"].update(accelerate_env_vars)
print ( ray_init_config["address"] )
# ray_init_config["runtime_env"]["working_dir"] = os.path.join("/home/ubuntu/llm-on-ray")
print("Start to init Ray connection")
context = ray.init(**ray_init_config)

def get_client_private_ipv4():
    # Get the hostname of the current machine
    hostname = socket.gethostname()
    # Get the IP address of the current machine
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def list_finetuned_models (finetuned_model_path, new_base_model_path):

    _all_models: Dict[str, FinetunedConfig] = {k: all_models[k] for k in sorted(all_models.keys())}
    _models: Dict[str, InferenceConfig] = {}

    if new_base_model_path and os.path.exists(new_base_model_path):
        for model_file in os.listdir(new_base_model_path):
            file_path = new_base_model_path + "/" + model_file
            if os.path.isdir(file_path):
                continue
            with open(file_path, "r") as f:
                m: InferenceConfig = parse_yaml_raw_as(InferenceConfig, f)
                _models[m.name] = m

        _all_models.update (_models)
        print ("update all model")
        print (_all_models)

    finetuned_models = {}
    for yaml_files in os.listdir(finetuned_model_path):

        if ".config.yaml" not in yaml_files: 
            continue
        yaml_data = yaml.safe_load (
            open(os.path.join (finetuned_model_path, yaml_files), "r")
            )
        # model_id_or_path = yaml_files.model_id_or_path
        # tokenizer_name_or_path = yaml_files.tokenizer_path
        # prompt = ""
        # chat_processor = 
        new_prompt = Prompt()
        new_prompt.intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        new_prompt.human_id = "\n### Instruction"
        new_prompt.bot_id = "\n### Response"
        new_prompt.stop_words.extend(
            ["### Instruction", "# Instruction", "### Question", "##", " ="]
        )

        new_model_desc = ModelDescription(
                    model_id_or_path=yaml_data['model_description'] ['model_id_or_path'],
                    tokenizer_name_or_path=yaml_data['model_description']['tokenizer_name_or_path'],
                    prompt=new_prompt,
                    chat_processor= yaml_data['model_description']['chat_processor'],
                )

        new_model_name = yaml_data ['name']
        new_finetuned = FinetunedConfig(
            name= new_model_name,
            route_prefix="/" + new_model_name,
            model_description=new_model_desc,
        )
        finetuned_models[new_model_name] = new_finetuned
        _all_models[new_model_name] = new_finetuned

    for model in _all_models:
        print (model)

    return _all_models

def deploy (
        model_name: str, 
        deploy_name: str,
        head_node_ip: str,
        finetuned_model_path: str = "",
        custom_model_path: str = "",
        replica_num: int = 1,
        cpus_per_worker_deploy: int = 3,
        hf_token: str = "",
        config_file: str = ""
    ):
    # self.deploy_stop()

    print("Deploying model:" + model_name)
    print ("custom model path:", custom_model_path)

    # process_tool = reset_process_tool( model_name )
    _all_models = list_finetuned_models (finetuned_model_path, custom_model_path) #update the all_model list with finetune models in user directory
    #for m in _all_models: print (m) 
    finetuned = _all_models[model_name]

    print ("model to deploy:", finetuned)

    # if cpus_per_worker_deploy * replica_num > int(ray.available_resources()["CPU"]):
    #     raise Exception("Resources are not meeting the demand")
    if config_file:
        with open (config_file, "r") as f:
            finetuned_deploy = parse_yaml_raw_as (InferenceConfig,f)
    else: 
        finetuned_deploy = finetuned.copy(deep=True)

    finetuned_deploy.name = deploy_name
    print ("deploy name:", finetuned_deploy.name)

    finetuned_deploy.route_prefix = f"/{finetuned_deploy.name}"
    print ("prefix:", finetuned_deploy.route_prefix)
    
    finetuned_deploy.device = "cpu"
    finetuned_deploy.ipex.precision = "bf16"
    finetuned_deploy.cpus_per_worker = cpus_per_worker_deploy
    finetuned_deploy.model_description.config.use_auth_token = hf_token

    print ("Inference Config:", finetuned_deploy)

    # transformers 4.35 is needed for neural-chat-7b-v3-1, will be fixed later
    if "neural-chat" in model_name:
        pip_env = "transformers==4.35.0"
    elif "fuyu-8b" in model_name:
        pip_env = "transformers==4.37.2"
    else:
        pip_env = "transformers==4.38.1"

    deployment = PredictorDeployment.options(  # type: ignore
        num_replicas=replica_num,
        ray_actor_options={
            "num_cpus": cpus_per_worker_deploy,
            "runtime_env": {"pip": [pip_env]},
        },
    ).bind(finetuned_deploy)

    # print ("ray serve status", serve.status())
    serve.run(
        deployment,
        _blocking=True,
        port=finetuned_deploy.port,
        name=finetuned_deploy.name,
        route_prefix=finetuned_deploy.route_prefix,
        host = "0.0.0.0"
    )
    # print ("ray serve status", serve.status())
    # head_node_ip = get_client_private_ipv4()
    endpoint = f"http://{head_node_ip}:{finetuned_deploy.port}/{finetuned_deploy.route_prefix}"

    print (serve.status())

    return endpoint, finetuned_deploy

def main ():
    parser = argparse.ArgumentParser(
        description="Deploy model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        default=None,
        help="Name of base/finetuned model"
    )
    parser.add_argument(
        "--deploy_name",
        type=str,
        required=True,
        default=None,
        help="Model serve endpoint prefix"
    )
    parser.add_argument(
        "--head_node_ip",
        type=str,
        required=True,
        default=None,
        help="ip of head node"
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        required=False,
        default=None,
        help="finetuned model path"
    )
    parser.add_argument (
        "--custom_model_path",
        type=str,
        required = False,
        default=None,
        help="custom model config path"
    )
    parser.add_argument(
        "--replica_num",
        type=int,
        required=False,
        default=None,
        help="Number of replica"
    )
    parser.add_argument(
        "--cpus_per_worker_deploy",
        type=int,
        required=False,
        default=None,
        help="Number of CPUs per replica"
    )
    parser.add_argument (
        "--huggingface_token",
        type=str,
        required=False,
        default="",
        help="Huggingface token to access gated model"
    )

    parser.add_argument (
        "--config_file",
        type=str,
        required=False,
        default = "",
    )
    args = parser.parse_args()
    model_name = args.model_name
    deploy_name = args.deploy_name
    head_node_ip = args.head_node_ip
    finetuned_model_path = args.finetuned_model_path
    custom_model_path = args.custom_model_path
    replica_num = args.replica_num
    cpus_per_worker_deploy = args.cpus_per_worker_deploy
    hf_token = args.huggingface_token
    config_file = args.config_file

    print ("deploying jobs..")
    deploy (
        model_name=model_name,
        deploy_name=deploy_name,
        head_node_ip=head_node_ip,
        finetuned_model_path=finetuned_model_path,
        custom_model_path=custom_model_path,
        replica_num=replica_num,
        cpus_per_worker_deploy=cpus_per_worker_deploy,
        hf_token= hf_token,
        config_file = config_file
    )

if __name__=="__main__":
    main()