import os
import json
import boto3
from botocore.config import Config
import logging
from typing import List, Callable, Optional
from urllib.parse import urlparse
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.llms.bedrock import Bedrock
from .fastapi_request import Request


logger = logging.getLogger(__name__)

class ContentHandlerForTextGeneration(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs" : [[{"role" : "system",
        "content" : "You are a Medical Knowledge Aware robot."},
        {"role" : "user", "content" : prompt}]],
        "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

def get_bedrock_client(
    region: Optional[str] = "us-east-1",
    runtime: Optional[bool] = True,
):
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client

def load_vector_db_faiss(vectordb_s3_path: str, vectordb_local_path: str, region: str) -> FAISS:
    os.makedirs(vectordb_local_path, exist_ok=True)
    # download the vectordb files from S3
    # note that the following code is only applicable to FAISS
    # would need to be enhanced to support other vector dbs
    vectordb_files = ["index.pkl", "index.faiss"]
    for vdb_file in vectordb_files:        
        s3 = boto3.client('s3')
        fpath = os.path.join(vectordb_local_path, vdb_file)
        with open(fpath, 'wb') as f:
            parsed = urlparse(vectordb_s3_path)
            bucket = parsed.netloc
            path =  os.path.join(parsed.path[1:], vdb_file)
            logger.info(f"going to download from bucket={bucket}, path={path}, to {fpath}")
            s3.download_fileobj(bucket, path, f)
            logger.info(f"after downloading from bucket={bucket}, path={path}, to {fpath}")

    # files are downloaded, lets load the vectordb
    logger.info("creating an embeddings object to hydrate the vector db")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_db = FAISS.load_local(vectordb_local_path, embeddings)
    logger.info(f"vector db hydrated, type={type(vector_db)} it has {vector_db.index.ntotal} embeddings")

    return vector_db
    
def load_vector_db_faiss_local() -> FAISS:
    # download the vectordb files from S3
    # note that the following code is only applicable to FAISS
    # would need to be enhanced to support other vector dbs
    logger.info("inside faiss local")

    DB_FAISS_PATH = './app/vectorstore/db_faiss'
    
    # files are downloaded, lets load the vectordb
    logger.info("creating an embeddings object to hydrate the vector db")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    logger.info(f"vector db hydrated, type={type(vector_db)} it has {vector_db.index.ntotal} embeddings")

    return vector_db

def setup_sagemaker_endpoint_for_text_generation(req: Request, region: str = "us-east-1") -> Callable:
    
    endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b-f"
    content_handler = ContentHandlerForTextGeneration()
    
    sm_llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,
        model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.6},
        endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
        content_handler=content_handler)
    
    return sm_llm

def setup_bedrock_endpoint(req: Request, region: str = "us-east-1") -> Callable:
    bedrock_client = get_bedrock_client(region=region, runtime=True)
    br_llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock_client, model_kwargs={'max_tokens_to_sample':4000})

    return br_llm
