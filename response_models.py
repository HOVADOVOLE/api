from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class NumberOfZeroCrossingsResponseModel(BaseModel):
    num_zero_crossings_down: int
    num_zero_crossings_up: int
class AverageDurationsResponseModel(BaseModel):
    average_ventilation_time: float
    average_increase_time: float
class IncreaseDecreaseResponseModel(BaseModel):
    co2_increase_count: int
    co2_decrease_count: int
class CO2DissipationResponseModel(BaseModel):
    average_dissipation_time_minutes: float
class CriticalSensor(BaseModel):
    device: str
    co2: float
class CriticalSensorsResponseModel(BaseModel):
    critical_sensors: List[CriticalSensor]
class DeviceModel(BaseModel):
    device: Optional[str] = None

class AnalyzeCO2ResponseModel(BaseModel):
    critical_devices: List[DeviceModel]
class TimeEstimate(BaseModel):
    time: datetime
    people_estimate: int
class PeopleEstimationResponse(BaseModel):
    time_estimates: List[TimeEstimate]
class SensorEstimation(BaseModel):
    time_estimates: List[TimeEstimate]
class SensorsEstimatesResponse(BaseModel):
    sensors_estimates: List[SensorEstimation]
class NotFoundModel(BaseModel):
    detail: str