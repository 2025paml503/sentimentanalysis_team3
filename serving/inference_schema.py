from typing import List

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 5000

class SentimentRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Review text",
    )
    title: str = Field(
        default="",
        max_length=500,
        description="Title",
    )

    @field_validator("review_text")
    @classmethod
    def must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Review text cannot be blank")
        return v

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Sentiment")
    positive_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence")
    oov_rate: float = Field(..., ge=0.0, le=0.0, description="Oov_rate")
    model_version: str
    latency_ms: float= Field(..., ge=0.0)

class BatchSentimentRequest(BaseModel):
    reviews: List[SentimentRequest] = Field(..., min_length=1, max_length=100)

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    count: int
    total_latency_ms: float