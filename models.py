# models.py
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    context: Optional[List[str]] = []

class QueryRequest(BaseModel):
    query: str