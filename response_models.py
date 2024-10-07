from pydantic import BaseModel
from typing import List, Optional, Dict, Union
from datetime import datetime, time

class CO2DissipationResponseModel(BaseModel):
    average_dissipation_time_minutes: float
class CriticalSensor(BaseModel):
    device: str
    co2: float
class DeviceModel(BaseModel):
    device: Optional[str] = None
class AnalyzeCO2ResponseModel(BaseModel):
    critical_devices: List[DeviceModel]
class TimeEstimate(BaseModel):
    time: datetime
    people_estimate: int
class SensorEstimation(BaseModel):
    time_estimates: List[TimeEstimate]
class ErrorModelResponse(BaseModel):
    detail: str
class TimeEstimates(BaseModel):
    time: time  # Jenom čas, bez datumu
    people_estimate: int
    co2_level: float  # Přidáváme CO2 úroveň do odpovědi

class PeopleEstimationsResponse(BaseModel):
    time_estimates: List[TimeEstimates]
class AverageDurationsResponse(BaseModel):
    average_ventilation_time: float
    average_increase_time: float
class IncreaseDecreaseResponse(BaseModel):
    co2_increase_count: float
    co2_decrease_count: float
class TrendPrediction(BaseModel):
    trend: str
    start_time: str
    end_time: str
class PredictionResponse(BaseModel):
    data: List[TrendPrediction]

class PeopleEstimationData(BaseModel):
    time: str
    people_count: int
class PeopleEstimationRM(BaseModel):
    time_estimates: List[TimeEstimate]

class DeviceData(BaseModel):
    device: str  # Identifikátor zařízení (čidlo)
    data: List[PeopleEstimationData]  # Seznam dat pro toto zařízení