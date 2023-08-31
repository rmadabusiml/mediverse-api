import os
import boto3
from enum import Enum
from pydantic import BaseModel

ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
REGION = boto3.Session().region_name

class Request(BaseModel):
    q: str
    max_length: int = 500
    num_return_sequences: int = 1
    top_k: int = 250
    top_p: float = 0.95
    do_sample: bool = False
    temperature: float = 1
    verbose: bool = False
    max_matching_docs: int = 3 
