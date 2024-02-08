from pydantic import BaseModel
from typing import List, Union

# Example Classes
class FineTuneModel(BaseModel):
    path: str
    hyperparam_1: int
    hyperparam_2: List[int]

class DeployModel(BaseModel):
    path: str
    hyperparam_1: int
    hyperparam_2: List[int]
