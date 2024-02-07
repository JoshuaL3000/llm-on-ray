#llm ray class wrapper 
'''
This is only a POC script to define and document the class and the list of functions
The subsequent development work will be carried on based on this document.

Please feel free to review, add, remove or modify the list of functions, descriptions, input parameters and output parameters.

'''

class llmray:
    def __init__( self, working_directory ):
        #define class input parameter
        self.working_directory = working_directory
        '''
        description: We assume to store and load model from the specified working directory

        input parameters
            working directory: str

        '''
    #################
    # Finetune
    #################

    def add_base_model (self, input_params):
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

    def finetune_start (self, input_params ):

        '''
        description: To start the finetuning job and save the trained model params at the working directory

        input parameters
            base_model: str
            dataset: str  -- data path
            hyper_parameter: dict
            output_model_name: str

        return
            json object: job info, status, etc
        '''
    
    def finetune_stop (self, input_params ):

        '''
        description: To cancel a started finetuning job gracefully

        input parameters
            job_id: str  --  assuming stopping by job id

        return
            json object: job info, status, etc
        '''

    def finetune_list_jobs (self, input_params ):

        '''
        description: To provide a list of finetune jobs with job status

        return
            json object: job info, status, etc
            example: {}

            
        '''

    ###############
    #Deploy
    ###############

    def deploy (self, input_params):
        '''
        description: To deploy a LLM model to a local endpoint at specific port. 
        (Please advice on LoRA implementation: how to load and store finetune weights and original weights)

        input parameters
            model_name/model_uuid: str
            port: str

        return
            endpoint: str
            deploy_id: str
        '''

    def deploy_stop (self, input_params):
        '''
        description: To kill an endpoint

        input parameters
            

        return
            status: str

        '''

    def endpoint_list (self):
        '''
        description: return a list of endpoints available with their info

        return
            json object list
        '''

    ##############
    # Chat
    #
    ##############
    def chatbot (self, input_params):
        '''
        (please advice on OpenAI-chatbot-like inference implementation)

        input parameters:
            session_hash
            endpoint

        return
    
        '''

    def generate_rag_vector_store (self, input_params):
        '''
        (please advice on this function: 
        
        Will we need another function to parse pdf/text/html/webpage into a certain format before ingesting for rag?
        )

        input parameters
            input_data: (format tbd)
            vector_store_name: str
        '''
    
    def chatbot_with_rag (self, input_params):
        '''
        (please advice on OpenAI-chatbot-like implementation and specify which vector store to use)


        '''

    # def send_message (self, input_params):
    #     '''
    #     description: send a chat message to a chat window
    #     (please advice on this)

    #     input parameters
    #         chatbot_id
    #         message
    #         session_hash

    #     return
    #         response
    #     '''

    ##############
    # Profiling
    ##############

    def resource_monitoring (self, input_params):
        '''
        description: a job profiler to profile the resource usage for all llm-ray jobs and overall system utilization
        (need advice: are we able to profile each single job (finetune/deploy/inference)?
        )

        
        '''
