from pydantic import BaseModel
from typing import List, Optional


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
    device: Optional[str] = None  # `device` může být None, pokud je prázdný objekt

class AnalyzeCO2ResponseModel(BaseModel):
    critical_devices: List[DeviceModel]