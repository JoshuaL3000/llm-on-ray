from wrapper.llmray_class import llmray
from wrapper.configs import finetune_config, ray_init_config
from inference.inference_config import all_models
import ray
from ray import serve
from finetune.finetune import get_accelerate_environment_variable

print ("All models:", all_models.keys())
accelerate_env_vars = get_accelerate_environment_variable(
        finetune_config["Training"]["accelerate_mode"], config=None
    )
ray_init_config["runtime_env"]["env_vars"].update(accelerate_env_vars)
print("Start to init Ray connection")

context = ray.init(**ray_init_config)

print("Ray connected")
head_node_ip = context.get("address").split(":")[0]
print ("head node ip:", head_node_ip)
print ("______________________________________________________")

serve.shutdown()
llm = llmray(
        all_models= all_models,
        base_models=all_models,
        working_dir= "/home/ubuntu/testdir",
        config=finetune_config,
        head_node_ip=head_node_ip,
        node_port= "22",
        node_user_name= "ubuntu",
        conda_env_name="llm-on-ray",
        master_ip_port=""    
    )


dm = llm.download_base_model ('opt-125m', base_model_location= "/home/ubuntu/base_models")

print ("download done")
print ("______________________________________________________")

base_model_ready = llm.check_base_model ("opt-125m")

print ("model exists: ", base_model_ready)
print ("______________________________________________________")

endpoint_1, deploy_obj_1 = llm.deploy (
    model_name="opt-125m", 
    replica_num=1, 
    cpus_per_worker_deploy=1)

print ("deploy success!")
print (endpoint_1)
print (deploy_obj_1)
print ("endpoint %s is up" % deploy_obj_1.name)
print ("model:", deploy_obj_1.model_description.model_id_or_path)

print ("test chatbot without rag")
response = llm.chatbot_rag (
    history= [["Who is the portfolio manager for JP Morgan management fund?", None]],
    model_endpoint=endpoint_1,
    Max_new_tokens=50,
    Temperature=0.35,
    Top_p=0.6,
    Top_k=0,
    rag_selector=False,
    rag_store_name="",
    returned_k = None
    # model_name
)

for r in response:
    print (r)

print ("______________________________________________________")

print ("deploying another endpoint 2")

endpoint_2, deploy_obj_2 = llm.deploy (
    model_name="bloom-560m", 
    replica_num=1, 
    cpus_per_worker_deploy=2)

print ("deploy success!")
print (endpoint_2)
print (deploy_obj_2)
print ("endpoint %s is up" % deploy_obj_2.name)
print ("model:", deploy_obj_2.model_description.model_id_or_path)

print ("test chatbot without rag")
response = llm.chatbot_rag (
    history= [["Who is the portfolio manager for JP Morgan management fund?", None]],
    model_endpoint=endpoint_2,
    Max_new_tokens=128,
    Temperature=0.35,
    Top_p=1,
    Top_k=0,
    rag_selector=False,
    rag_store_name="",
    returned_k = 1
    # model_name
)

for r in response:
    print (r)

print ("______________________________________________________")

print ("creating rag from pdf data")
db_dir = llm.generate_rag_vector_store (
    rag_store_name="jvasx",
    upload_type="",
    input_type="local",
    input_texts="/home/ubuntu/testdir/testpdf",
    depth= 1,
    upload_files= "",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    splitter_chunk_size=500,
    cpus_per_worker=1
)
print ("Data RAG at ", db_dir)

print ("______________________________________________________")

print ("test chatbot with rag")
response = llm.chatbot_rag (
    history= [["Who is the portfolio manager for JP Morgan management fund?", None]],
    model_endpoint=endpoint_2,
    Max_new_tokens=128,
    Temperature=0.35,
    Top_p=1,
    Top_k=0,
    rag_selector=True,
    rag_store_name="jvasx",
    returned_k = 1,
    # model_name
)

for r in response:
    print (r)

print ("______________________________________________________")


print ("deleting endpoint ", deploy_obj_1.name)
llm.deploy_stop (deploy_obj_1.name)

print ("deleting endpoint ", deploy_obj_2.name)
llm.deploy_stop (deploy_obj_2.name)
