#llm ray class wrapper 
'''
This is only a POC script to define and document the class and the list of functions
The subsequent development work will be carried on based on this document.

Please feel free to review, add, remove or modify the list of functions, descriptions, input parameters and output parameters.

'''

# from model import FineTuneModel, DeployModel

import requests
import time
import os
import sys
import yaml
import requests
import datetime
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.inference_config import all_models, ModelDescription, Prompt
# print (all_models)
from inference.inference_config import InferenceConfig as FinetunedConfig
# from inference.chat_process import ChatModelGptJ, ChatModelLLama, ChatModelwithImage  # noqa: F401
from wrapper.utils import is_simple_api, history_to_messages, add_knowledge, ray_status_parser

import huggingface_hub
import transformers
from ray import serve
import ray
from typing import Dict, List, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# from pyrecdp.primitives.document.reader import _default_file_readers
from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE

if ("RECDP_CACHE_HOME" not in os.environ) or (not os.environ["RECDP_CACHE_HOME"]):
    os.environ["RECDP_CACHE_HOME"] = "/tmp/"

class llmray:
    # def __init__( self, working_directory ):
        #define class input parameter
        # self.working_directory = working_directory
    '''
    description: We assume to store and load model from the specified working directory

    input parameters_base_models
        working directory: str

    '''
    def __init__(
        self,
        all_models: Dict[str, FinetunedConfig],
        base_models: Dict[str, FinetunedConfig],
        working_dir: str,
        config: dict,
        head_node_ip: str,
    ):
        self._all_models = all_models
        self._base_models = base_models

        #setting all the paths
        self.working_dir = working_dir
        # self.base_model_path = "/base_models/"  #not using now
        self.finetuned_model_path = os.path.join (working_dir, "finetuned_models") #everytime after a model is finetuned, we need to output the yaml config to this directory for list models to load.
        # self.finetuned_checkpoint_path = os.path.join (working_dir, "finetuned_checkpoint") #finetune
        #self.default_data_path = os.path.join (working_dir, "data_path/data.jsonl") #finetune
        self.default_rag_store_path = os.path.join (working_dir, "rag_vector_stores") #later should be change to argument of regenerate function

        #be in this way for now, change later
        file_path = os.path.abspath(__file__)
        infer_path = os.path.dirname(file_path)
        self.repo_code_path = os.path.abspath(infer_path + os.path.sep + "../")
        
        #set ip and port
        self.head_node_ip = head_node_ip
        # self.node_port = node_port
        # self.master_ip_port = master_ip_port
        self.ip_port = "http://127.0.0.1:8000" #this is deploy endpoint ip 

        #ray configs
        self.config = config
        self.test_replica = 4
        self.bot_queue = list(range(self.test_replica))
        self.messages = [
            "What is AI?",
            "What is Spark?",
            "What is Ray?",
            "What is chatbot?",
        ]
        self.process_tool = None
        # self.finetune_actor = None
        # self.finetune_status = False
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.all_finetune_jobs = []
        self.all_deployed_endpoints = []

    #################
    # Finetune
    #################

    def add_base_model (self, input_params):
        #grey out first in UI
        '''
        description: user can add a new base model from huggingface, base model can be shared by different users

        input parameters:
            model_name: str
            model_id: str
            tokenizer_id: str
            huggingface_token: str
            config: dict

        return:
           json object
        '''

    def download_base_model(self, model_name, base_model_location, token:str=None): #assume base models are saved to .cache/huggingface/hub
        
        if token:
            huggingface_hub.login (token = token)

        if model_name not in self._base_models:
            raise Exception ("Model not found in Base model list")
        
        model_desc = self._base_models [model_name]
        print (model_desc)
        tokenizer_name_or_path = model_desc.model_description.tokenizer_name_or_path
        model_id_or_path = model_desc.model_description.model_id_or_path
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id_or_path
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained (
                tokenizer_name_or_path
            )

            #save the model
            model_location = os.path.join ( base_model_location, model_name )
            model.save_pretrained (model_location)

            #save tokenizer
            tokenizer.save_pretrained (model_location)

            #rewrite the location in base model yaml
            import yaml

            _cur = os.path.dirname(os.path.abspath(__file__))
            _models_folder = _cur + "/../inference/models"
        
            file_path = _models_folder + "/" + model_name + ".yaml"
            print ("******************")
            print (file_path)
            if os.path.isfile(file_path):
                with open(file_path, "r") as f:                   
                    model_data = yaml.safe_load (f)
                model_data ["model_description"]["model_id_or_path"] = model_location
                model_data ["model_description"]["tokenizer_name_or_path"] = model_location

                with open(file_path, "w") as f:  
                    yaml.dump (model_data, f)
            else:
                raise Exception ("Model Yaml not found.")

            #update _all_models and _base_models
            self._base_models[model_name].model_description.model_id_or_path = model_location
            self._base_models[model_name].model_description.tokenizer_name_or_path = model_location

            self._all_models[model_name].model_description.model_id_or_path = model_location
            self._all_models[model_name].model_description.tokenizer_name_or_path = model_location

            print ( self._base_models[model_name].model_description )
            print ( self._all_models[model_name].model_description )

        except Exception as err: 
            print ( "Download fail", str (err ))

    def add_finetuned_model (self, model_name, finetuned_model_location):
        """
        create yaml file
        update _all_model and _finetuned_model
        
        """

    def check_local_base_model (self, model_name):
        # check the yaml file to see the path is correct
        # if model id or tokenizer id path not found or not tally, will return error
        model_exist = False

        model_desc = self._base_models [model_name]
        print (model_desc)

        # tokenizer_name_or_path = model_desc.model_description.tokenizer_name_or_path
        model_id_or_path = model_desc.model_description.model_id_or_path
        tokenizer_name_or_path = model_desc.model_description.tokenizer_name_or_path

        import yaml
        _cur = os.path.dirname(os.path.abspath(__file__))
        _models_folder = _cur + "/../inference/models"
    
        file_path = _models_folder + "/" + model_name + ".yaml"
 
        if os.path.isfile(file_path):
            print ("yaml path exist")
            with open(file_path, "r") as f:                   
                model_data = yaml.safe_load (f)
        # check base model location whether model exist
                
            if model_data ["model_description"]["model_id_or_path"] == model_id_or_path and \
                model_data ["model_description"]["tokenizer_name_or_path"] == tokenizer_name_or_path:

                print ("model yaml config tally with model dict")
                print (model_id_or_path)

                if os.path.isdir (model_id_or_path) and os.path.isfile ( model_id_or_path + "/config.json"):
                    
                    print ("Base model exists in local")
                    model_exist = True

        return model_exist

        # check _all_model and _base_model dict
    
    def job_submission_client ( #using this way to avoid double ray init issue
            self,
            submission_id:str, 
            entrypoint: str, #basically the command to run the job. the files must be present in the head
            # head_node_ip:str = "", 
            port:str = "8265", 
            runtime_env: dict = {}, #if already defined in your script in entrypoint, no need to define here
            metadata= {} #you can set custom metadata as remarks or something
        ): 

        # if not head_node_ip:
        head_node_ip = self.head_node_ip

        url = "http://" + head_node_ip + ":" + port + "/api/jobs/"
        print (url)
        resp = requests.post (
            url = url,
            json = {
                "entrypoint" : entrypoint,
                "submission_id" : submission_id,
                "runtime_env" : runtime_env,
                # "metadata" : metadata
            }
        )

        print (resp.content)
        print ("resp:", resp.json())

        if resp.status_code == 200:
            return {"submission_id": submission_id, "message": "SUCCESS"}
        else:
            return {"submission_id": None, "message": "FAIL"}
    
    def finetune_start(
            self,
            dataset_path,
            new_model_name,
            model_name, #base_model
            # custom_model_name, #not supporting this for now
            batch_size,
            num_epochs,
            max_train_step,
            lr,
            worker_num,
            cpus_per_worker_ftn,
        ):
        # if model_name == "specify other models":
        #     model_desc = None
        #     origin_model_path = custom_model_name
        #     if "gpt" in model_name.lower() or "pythia" in model_name.lower():
        #         gpt_base_model = True
        #     else:
        #         gpt_base_model = False
        # else:
        
        # head_dataset_path = os.path.join("/home/ubuntu/llm-on-ray/datasets", os.path.basename (dataset_path) )
        model_desc = self._base_models[model_name].model_description
        print ("model_desc:",model_desc)
        origin_model_path = model_desc.model_id_or_path
        gpt_base_model = model_desc.gpt_base_model

        # last_gpt_base_model = False
        finetuned_model_path = os.path.join(self.finetuned_model_path, model_name, new_model_name)
        finetuned_checkpoint_path = os.path.join(self.finetuned_model_path, model_name, new_model_name + "-checkpoint")

        finetune_config = self.config.copy()       
        
        finetune_config["Dataset"]["train_file"] = dataset_path
        finetune_config["Dataset"]["validation_file"] = None
        finetune_config["Dataset"]["validation_split_precentage"] = 0

        finetune_config["General"]["base_model"] = origin_model_path
        if finetuned_checkpoint_path:
            finetune_config["General"]["checkpoint_dir"] = finetuned_checkpoint_path 
        finetune_config["General"]["config"]["trust_remote_code"] = True
        finetune_config["General"]["config"]["use_auth_token"] = None
        finetune_config["General"]["gpt_base_model"] = gpt_base_model
        finetune_config["General"]["output_dir"] = finetuned_model_path

        finetune_config["Training"]["accelerate_mode"] = "CPU_DDP" #only support CPU now
        finetune_config["Training"]["batch_size"] = batch_size
        finetune_config["Training"]["device"] = "CPU"
        finetune_config["Training"]["epochs"] = num_epochs
        finetune_config["Training"]["learning_rate"] = lr
        finetune_config["Training"]["lr_scheduler"] = "linear"     
        if max_train_step != 0:
            finetune_config["Training"]["max_train_steps"] = max_train_step
        finetune_config["Training"]["num_training_workers"] = worker_num
        finetune_config["Training"]["optimizer"] = "AdamW"
        finetune_config["Training"]["resources_per_worker"]['CPU'] = cpus_per_worker_ftn
        finetune_config["Training"]["weight_decay"] = 0.0

        finetune_config["failure_config"]['max_failures'] = 4

        config_file_name = os.path.join (self.finetuned_model_path, new_model_name + ".yaml" )
        with open(config_file_name, "w") as f:
            yaml.dump(finetune_config, f)

        #make inference config yaml
        inference_config = {}
        inference_config ['port'] = 8000
        inference_config ['name'] = new_model_name
        inference_config ['route_prefix'] = ''
        inference_config ['num_replicas'] = 1
        inference_config ['cpus_per_worker'] = 16
        inference_config ['gpus_per_worker'] = 0
        inference_config ['deepspeed'] = False
        inference_config ['workers_per_group'] = 1
        inference_config ['device'] = "cpu"
        inference_config ['ipex'] = {}
        inference_config ['ipex']['enabled'] = False
        inference_config ['ipex']['precision'] = 'bf16'
        inference_config ['model_description'] = {}
        inference_config ['model_description']['model_id_or_path'] = finetuned_model_path
        inference_config ['model_description']['tokenizer_name_or_path'] = model_desc.tokenizer_name_or_path
        inference_config ['model_description']['chat_processor'] = model_desc.chat_processor

        inference_config_file_name = os.path.join (self.finetuned_model_path, new_model_name + ".config.yaml" )
        with open(inference_config_file_name, "w") as f:
            yaml.dump(inference_config, f)

        job_submission_status = self.job_submission_client (
            submission_id = f"{new_model_name}-{int(datetime.datetime.now().timestamp())}",
            entrypoint=f"cd /home/ubuntu/llm-on-ray; python finetune/finetune.py --config_file {config_file_name}",
            metadata = finetune_config
        )

        submission_id = job_submission_status ['submission_id']
        if job_submission_status ['message'] == "FAIL":
            raise RuntimeError ("Job submission unsuccessful")     
        
        self.all_finetune_jobs.append (submission_id)
        return submission_id
    
    def finetune_stop (
            self, 
            submission_id: str,
            port: str = "8265"
        ):
        url = f"http://{self.head_node_ip}:{port}/api/jobs/{submission_id}/stop"
        resp = requests.post (url)

        if resp.status_code == 200:
            return resp.json () ['stopped']
        else:
            raise ConnectionRefusedError (f"Unable to stop job with status code {resp.status_code}: {str(resp.content)}")

    #return all finetune job metadata and status
    def finetune_list_jobs (
            self,
            port:str = "8265" 
        ):
        url = "http://" + self.head_node_ip + ":" + port + "/api/jobs/"
        data = requests.get (url)
        if data.status_code == 200:
            data_json = data.json()
            job_list = []
            for job in data_json:
                # if job ['submission_id'] in self.all_finetune_jobs:
                job_list.append (job)
            return job_list
        else:
            raise ConnectionError (f"Getting {str(data.status_code)} response from GET requests. \n {str(data.content)}")

    #return single job metadata and status
    def get_job_status (
            self, 
            submission_id: str,
            port: str = "8265"
        ):
        url = "http://" + self.head_node_ip + ":" + port + f"/api/jobs/{submission_id}"
        data = requests.get (url)
        if data.status_code == 200:
            status = data.json()['status']
            return status
        else:
            raise ConnectionError (f"Getting {str(data.status_code)} response from GET requests. \n {str(data.content)}")

    def remove_finetuned_models (
        self,
        model_name:str,
        # base_model_name: str
    ):
        config_file_name = os.path.join (self.finetuned_model_path, model_name + ".yaml" )
        #check for base model name
        if os.path.isfile(config_file_name):
            print ("yaml path exist")
            with open(config_file_name, "r") as f:                   
                model_desc = yaml.safe_load (f)
        else:
            print ("Model config not found.")
            return False
        
        finetuned_model_path = model_desc ['General']['output_dir']
        finetuned_checkpoint_path = model_desc ['General']['checkpoint_dir']

        #delete from local
        try:
            shutil.rmtree (finetuned_model_path)
            shutil.rmtree (finetuned_checkpoint_path)
            os.remove (config_file_name)
            return True

        except Exception as err:
            print ("Unable to remove :", str(err))
            return False
        
    ###############
    #Deploy
    ###############
    def list_finetuned_models (self):

        finetuned_models = {}
        for yaml_files in os.listdir(self.finetuned_model_path):

            if ".config.yaml" not in yaml_files: 
                continue
            yaml_data = yaml.safe_load (
                open(os.path.join (self.finetuned_model_path, yaml_files), "r")
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
            self._all_models[new_model_name] = new_finetuned

        for model in self._all_models:
            print (model)

        return finetuned_models
         
    def reset_process_tool (self, model_name):
        self.list_finetuned_models () #update the all_model list with finetune models in user directory
        finetuned = self._all_models[model_name]
        model_desc = finetuned.model_description
        prompt = model_desc.prompt
        if model_desc.chat_processor is not None:
            chat_model = getattr(sys.modules[__name__], model_desc.chat_processor, None)
            if chat_model is None:
                return (
                    model_name
                    + " deployment failed. "
                    + model_desc.chat_processor
                    + " does not exist."
                )
            self.process_tool = chat_model(**prompt.dict())

    def deploy (
            self, 
            model_name: str, 
            deploy_name: str,
            replica_num: int = 1,
            cpus_per_worker_deploy: int = 3,
            hf_token: str = "",
        ):

        if hf_token: 
            entrypoint = f"cd /home/ubuntu/llm-on-ray/wrapper; python deploy_start.py --model_name {model_name} --deploy_name {deploy_name} --head_node_ip {self.head_node_ip} --finetuned_model_path {self.finetuned_model_path} --replica_num {replica_num} --cpus_per_worker_deploy {cpus_per_worker_deploy} --huggingface_token {hf_token}"
        else:
            f"cd /home/ubuntu/llm-on-ray/wrapper; python deploy_start.py --model_name {model_name} --deploy_name {deploy_name} --head_node_ip {self.head_node_ip} --finetuned_model_path {self.finetuned_model_path} --replica_num {replica_num} --cpus_per_worker_deploy {cpus_per_worker_deploy}" 

        job_submission_status = self.job_submission_client (
            submission_id = f"deploy-{int(datetime.datetime.now().timestamp())}",
            entrypoint= entrypoint,
            metadata = {}
        )
        submission_id = job_submission_status ['submission_id']
        if job_submission_status ['message'] == "FAIL":
            raise RuntimeError ("Job submission unsuccessful")   
        
        return submission_id
    
    def deploy_stop (self, deploy_name):
        '''
        description: To kill an endpoint

        input parameters
            

        return
            status: str

        '''
        serve.delete (deploy_name)
        print ("endpoint deleted.")
        #serve.delete (name)

    def endpoint_list (self):
        '''
        description: return a list of endpoints available with their info

        return
            json object list
        '''
        return serve.status()

    ##############
    # Chat
    #
    ##############
    # def
        
    def model_generate(self, prompt, request_url, model_name, config, simple_api=True):

        self.reset_process_tool (model_name)
        if simple_api:
            prompt = self.process_tool.get_prompt(prompt)
            sample_input = {"text": prompt, "config": config, "stream": True}
        else:
            sample_input = {
                "model": model_name,
                "messages": prompt,
                "stream": True,
                "max_tokens": config["max_new_tokens"],
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "top_k": config["top_k"],
            }
        proxies = {"http": None, "https": None}
        outputs = requests.post(request_url, proxies=proxies, json=sample_input, stream=True)
        outputs.raise_for_status()
        for output in outputs.iter_lines(chunk_size=None, decode_unicode=True):
            # remove context
            if simple_api:
                if prompt in output:
                    output = output[len(prompt) :]
            else:
                if output is None or output == "":
                    continue
                import json
                import re

                chunk_data = re.sub("^data: ", "", output)
                if chunk_data != "[DONE]":
                    # Get message choices from data
                    choices = json.loads(chunk_data)["choices"]
                    # Pick content from first choice
                    output = choices[0]["delta"].get("content", "")
                else:
                    output = ""
            yield output

    def chatbot (
        self,
        history,
        # deploy_model_endpoint,
        model_endpoint,
        Max_new_tokens,
        Temperature,
        Top_p,
        Top_k,
        model_name=None,
        image=None,
        enhance_knowledge=None,
    ):
        print ("**********history**********")
        print (history)
        # request_url = model_endpoint if model_endpoint != "" else deploy_model_endpoint
        request_url = model_endpoint

        # simple_api = is_simple_api(request_url, model_name)
        # if simple_api and image is not None:
        #     raise gr.Error("SimpleAPI image inference is not implemented.")
        simple_api = True
        
        prompt = history_to_messages(history, image)
        if enhance_knowledge:
            prompt = add_knowledge(prompt, enhance_knowledge)
            print ("----------------------------")
            print ("prompt for rag:", prompt)

        time_start = time.time()
        token_num = 0
        config = {
            "max_new_tokens": Max_new_tokens,
            "temperature": Temperature,
            "do_sample": True,
            "top_p": Top_p,
            "top_k": Top_k,
            "model": None,
        }
        outputs = self.model_generate(
            prompt=prompt,
            request_url=request_url,
            model_name=model_name,
            config=config,
            simple_api=simple_api,
        )

        if history[-1][1] is None:
            history[-1][1] = ""
        for output in outputs:
            if len(output) != 0:
                time_end = time.time()
                if simple_api:
                    history[-1][1] += output
                    history[-1][1] = self.process_tool.convert_output(history[-1][1])
                else:
                    history[-1][1] += output
                time_spend = round(time_end - time_start, 3)
                token_num += 1
                new_token_latency = f"""
                                    | <!-- --> | <!-- --> |
                                    |---|---|
                                    | Total Latency [s] | {time_spend} |
                                    | Tokens | {token_num} |"""
                yield [history, new_token_latency]

    def chatbot_rag (self,       
        history,
        # deploy_model_endpoint,
        model_endpoint,
        # process_tool_obj,
        Max_new_tokens,
        Temperature,
        Top_p,
        Top_k,
        rag_selector,
        rag_store_name,
        returned_k,
        model_name=None,
        image=None,
    ):
        
        enhance_knowledge = None

        if rag_selector:
            rag_path = os.path.join (self.default_rag_store_path, rag_store_name)
            if not os.path.isdir (rag_path):
                print (rag_path)
                raise Exception ("RAG vector store not found")

            if os.path.isabs(rag_path):
                tmp_folder = os.getcwd()
                load_dir = os.path.join(tmp_folder, rag_path)
            else:
                load_dir = rag_path
            if not os.path.exists(load_dir):
                raise Exception("The specified path does not exist")

            question = history[-1][0]
            print("history: ", history)
            print("question: ", question)

            if not hasattr(self, "embeddings"):
                local_embedding_model_path = os.path.join(
                    RECDP_MODELS_CACHE, self.embedding_model_name
                )
                if os.path.exists(local_embedding_model_path):
                    self.embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
                else:
                    self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

            vectorstore = FAISS.load_local(load_dir, self.embeddings, index_name="knowledge_db")
            sim_res = vectorstore.similarity_search(question, k=int(returned_k))
            enhance_knowledge = ""
            for doc in sim_res:
                enhance_knowledge = enhance_knowledge + doc.page_content + ". "

        bot_generator = self.chatbot(
            history,
            # deploy_model_endpoint,
            model_endpoint,
            Max_new_tokens,
            Temperature,
            Top_p,
            Top_k,
            model_name=model_name,
            image=image,
            enhance_knowledge=enhance_knowledge,
        )
        for output in bot_generator:
            yield output

    def generate_rag_vector_store (
        self,
        vector_store_name,
        data_path
    ):
        job_submission_status = self.job_submission_client (
            submission_id = f"rag-{int(datetime.datetime.now().timestamp())}",
            entrypoint=f"cd /home/ubuntu/llm-on-ray/wrapper; python rag.py --rag_store_name {vector_store_name} --data_path {data_path}",
            metadata = {}
        )

        submission_id = job_submission_status ['submission_id']
        if job_submission_status ['message'] == "FAIL":
            raise RuntimeError ("Job submission unsuccessful")   

        return submission_id
    
    ##############
    # Profiling
    ##############

    def resource_monitoring (self):
        return ray_status_parser (self.head_node_ip), ray.available_resources(),  ray.cluster_resources()
