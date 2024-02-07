#a dummy API used to showcase how the llm ray class is implemented

import LLMRay
 
@router.post('/finetune')
@decorator
async def finetune(dict=Body(...)):
    llmray_class = LLMRay()
    finetune_model = FineTuneModel(model_output_path="", hyperparam_1=1, hyperparam_2=2)
    response = llmray_class.finetune(FineTuneModel)
    return response
 