from pyrecdp.primitives.operations import *
from pyrecdp.LLM import TextPipeline

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
import os, sys, argparse

if ("RECDP_CACHE_HOME" not in os.environ) or (not os.environ["RECDP_CACHE_HOME"]):
    os.environ["RECDP_CACHE_HOME"] = "/tmp/"

def generate_rag_vector_store (
        rag_store_name,
        target_folder,
        embedding_model = "sentence-transformers/all-mpnet-base-v2",
        splitter_chunk_size = 500,
        cpus_per_worker = 2,
    ):
    db_dir = os.path.join (rag_store_name)
    loader = DirectoryLoader(input_dir=target_folder)

    if os.path.isabs(db_dir):
        tmp_folder = os.getcwd()
        save_dir = os.path.join(tmp_folder, db_dir)
    else:
        save_dir = db_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    vector_store_type = "FAISS"
    index_name = "knowledge_db"
    text_splitter = "RecursiveCharacterTextSplitter"
    splitter_chunk_size = int(splitter_chunk_size)
    text_splitter_args = {
        "chunk_size": splitter_chunk_size,
        "chunk_overlap": 0,
        "separators": ["\n\n", "\n", " ", ""],
    }
    embeddings_type = "HuggingFaceEmbeddings"

    embedding_model_name = embedding_model
    local_embedding_model_path = os.path.join(RECDP_MODELS_CACHE, embedding_model_name)
    if os.path.exists(local_embedding_model_path):
        embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
        embeddings_args = {"model_name": local_embedding_model_path}
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        embeddings_args = {"model_name": embedding_model_name}
        
    pipeline = TextPipeline()
    ops = [loader]

    ops.extend(
        [
            RAGTextFix(re_sentence=True),
            DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
            DocumentIngestion(
                vector_store=vector_store_type,
                vector_store_args={"output_dir": save_dir, "index": index_name},
                embeddings=embeddings_type,
                embeddings_args=embeddings_args,
                num_cpus=cpus_per_worker,
            ),
        ]
    )
    pipeline.add_operations(ops)
    pipeline.execute()
    
def main ():
    parser = argparse.ArgumentParser(
        description="Run RAG"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        default=None,
        help="Directory of the data folder"
    )
    parser.add_argument(
        "--rag_store_name",
        type=str,
        required=True,
        default=None,
        help="RAG STORE PATH"
    )
    args = parser.parse_args()
    data_path = args.data_path
    rag_store_name = args.rag_store_name

    generate_rag_vector_store (
        rag_store_name=rag_store_name,
        target_folder=data_path,
    )

if __name__=="__main__":
    main()
