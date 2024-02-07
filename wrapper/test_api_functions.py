#a dummy API used to showcase how the llm ray class is implemented
#modify this document accordingly in respect to llmray class

'''
To support multi-tenency and multi project FE application

multi-tenancy can be achieved by setting different work directory for different user to store their finetuned model weights and rag vector store

'''

import LLMRay
from easydata_util import *
from data_model import * #pydantic validation model
 

# Finetune
@router.post('/finetune_start')
@decorator
async def finetune_start(dict=Body(...)):
    user_id = dict.get ("user_id")
    project_id = dict.get ("project_id")
    workflow_id = dict.get ("workflow_id")
    model_name = dict.get ("model_name")
    work_dir = get_workdir (user_id)
    llmray_class = LLMRay( work_dir )
    # finetune_model = FineTuneModel(model_output_path="", hyperparam_1=1, hyperparam_2=2)
    response = llmray_class.finetune_start(params)

    upsert_job_status (user_id, model_name)
    return response
 
@router.post('/finetune_stop')
@decorator
async def finetune_stop(dict=Body(...)):

    job_name = dict.get ("job_name")
    user_id = dict.get ("user_id")
    work_dir = get_workdir (user_id)
    llmray_class = LLMRay( work_dir )
    job_id = get_job_id (job_name, user_id)

    response = llmray_class.finetune_stop(job_id = job_id, input_params = params )
    return response

@router.get('/finetune_list_job')
@decorator
async def finetune_list_job(dict=Body(...)):
    user_id = dict.get ("user_id")
    work_dir = get_workdir (user_id)
    llmray_class = LLMRay( work_dir )

    response = llmray_class.finetune_list_jobs( )
    return response

#deploy
@router.get('/deploy')
@decorator
async def deploy_job(dict=Body(...)):
    user_id = dict.get ("user_id")
    model_name = dict.get ("model_name")

    work_dir = get_workdir (user_id)
    port_number = get_free_port ()
    
    llmray_class = LLMRay( work_dir )

    response = llmray_class.deploy( model_name, port_number )
    upsert_deploy_status (model_name, port_number )

    return response

#chat / inference
@router.get('/chatbot')
@decorator
async def chatbot(dict=Body(...)):
    user_id = dict.get ("user_id")
    deploy_id = dict.get ("deploy_id")
    msg = dict.get ("message")

    endpoint = get_endpoint (deploy_id)
    session_hash = generate_session_hash ()

    work_dir = get_workdir (user_id)
    
    llmray_class = LLMRay( work_dir )
    response = llmray_class.chatbot( msg, endpoint, session_hash )

    return response