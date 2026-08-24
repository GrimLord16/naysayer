from pydantic import BaseModel


class PredictRequest(BaseModel):
    title: str
    body: str = ""


class LabelScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    alias: str
    model_version: str
    labels: list[LabelScore]


class ModelStatus(BaseModel):
    loaded: bool
    version: str | None = None


class HealthResponse(BaseModel):
    status: str
    champion: ModelStatus
    challenger: ModelStatus
