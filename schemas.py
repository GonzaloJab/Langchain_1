from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(
        default_factory=list,
        description="The sources used to answer the question"
        )

