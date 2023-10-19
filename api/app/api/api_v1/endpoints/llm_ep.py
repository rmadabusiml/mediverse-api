import os
import sys
import boto3
import logging
from typing import Any, Dict
from fastapi import APIRouter
from urllib.parse import urlparse
from langchain import PromptTemplate
from .fastapi_request import Request
from langchain.chains import RetrievalQA

from .initialize import (setup_bedrock_endpoint, setup_sagemaker_endpoint_for_text_generation,
                        load_vector_db_faiss_local)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()
#logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO)

# initialize the vector db as a global variable so that it
# can persist across lambda invocations
VECTOR_DB_DIR = os.path.join("/tmp", "_vectordb")
_vector_db = None
_current_vectordb_type = None
#_sm_llm = None
_br_llm = None
_qa_chain = None

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

custom_prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:
"""

router = APIRouter()

# def _init(req: Request):
#     # vector db is a global static variable, so that it only
#     # created once across multiple lambda invocations, if possible
#     global _vector_db
#     global _current_vectordb_type
#     #vectordb_s3_path = "s3://dml-dev-deployment/raghu/health-universe/db_faiss/"
    
#     if _vector_db is None:
#         # _vector_db = load_vector_db_faiss(vectordb_s3_path,
#         #                                   VECTOR_DB_DIR,
#         #                                   boto3.Session().region_name)
#         _vector_db = load_vector_db_faiss_local()
#         logger.info("after creating vector db")
#     elif _vector_db is not None:
#         logger.info(f"seems like vector db already exists...")

#     # just like the vector db the sagemaker endpoint used for
#     # text generation is also global and shared across invocations
#     # if possible
#     global _sm_llm
#     if _sm_llm is None:
#         logger.info(f"SM LLM endpoint is not setup, setting it up now")
#         _sm_llm = setup_sagemaker_endpoint_for_text_generation(req, boto3.Session().region_name)
#         logger.info("after setting up sagemaker llm endpoint")
#     else:
#         logger.info(f"sagemaker llm endpoint already exists..")
        
#     global _qa_chain
#     if _qa_chain is None:
#         logger.info(f"Retrieval QA Chain is not setup, setting it up now")
#         qa_prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
                            
#         _qa_chain = retrieval_qa_chain(_sm_llm, qa_prompt, _vector_db)
#         logger.info("after setting up Retrieval QA Chain")
#     else:
#         logger.info(f"Retrieval QA Chain already exists..")


def _init(req: Request):
    # vector db is a global static variable, so that it only
    # created once across multiple lambda invocations, if possible
    global _vector_db
    global _current_vectordb_type
    
    if _vector_db is None:
        # _vector_db = load_vector_db_faiss(vectordb_s3_path,
        #                                   VECTOR_DB_DIR,
        #                                   boto3.Session().region_name)
        _vector_db = load_vector_db_faiss_local()
        logger.info("after creating vector db")
    elif _vector_db is not None:
        logger.info(f"seems like vector db already exists...")

    # just like the vector db the Bedrock endpoint used for
    # text generation is also global and shared across invocations
    # if possible
    global _br_llm
    if _br_llm is None:
        logger.info(f"Bedrock LLM endpoint is not setup, setting it up now")
        #_sm_llm = setup_sagemaker_endpoint_for_text_generation(req, boto3.Session().region_name)
        _br_llm = setup_bedrock_endpoint(req, boto3.Session().region_name)
        logger.info("after setting up Bedrock llm endpoint")
    else:
        logger.info(f"Bedrock llm endpoint already exists..")
        
    global _qa_chain
    if _qa_chain is None:
        logger.info(f"Retrieval QA Chain is not setup, setting it up now")
        qa_prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
                            
        #_qa_chain = retrieval_qa_chain(_sm_llm, qa_prompt, _vector_db)
        _qa_chain = retrieval_qa_chain(_br_llm, qa_prompt, _vector_db)
        logger.info("after setting up Retrieval QA Chain")
    else:
        logger.info(f"Retrieval QA Chain already exists..")
        
        
#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    print("PROMPT: ", prompt)
    print("---------------------")
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_type="similarity", search_kwargs={'k': 4}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

    return qa_chain

@router.post("/rag")
async def rag_handler(req: Request) -> Dict[str, Any]:
    # dump the received request for debugging purposes
    logger.info(boto3.__version__)
    logger.info(f"req={req}")

    # initialize vector db and Sagemaker Endpoint
    _init(req)

    response = _qa_chain({"query": req.q})
    logger.info(response)
    
    answer = response["result"]
    sources = response["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
        
    resp = {'question': req.q, 'answer': answer}

    return resp

