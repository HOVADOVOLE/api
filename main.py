import os
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from dotenv import load_dotenv
import httpx
import pandas as pd
from fastapi.responses import JSONResponse
import response_models
import numpy as np

# Načtení .env.local souboru
load_dotenv('.env.local')

app = FastAPI()

# Načtení API URL a Secret Key z .env.local souboru
API_URL: str = os.getenv("VITE_API_URL")
SECRET_KEY: str = os.getenv("VITE_SECRET_KEY")
NORMAL_CO2_LEVEL: int = 400

headers = {
    "secret_key": SECRET_KEY,
}

# Povolení CORS pro všechny zdroje
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/records/number_of_zero_crossings/place/{device}/", response_model=response_models.NumberOfZeroCrossingsResponseModel)
async def get_number_of_zero_crossings(
    device: str = Path(..., description="Název zařízení", example="eui-70b3d57ed006209f-co-05"),
    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00")
):
    print(device, start, stop)
    endpoint = f"{API_URL}/records/place/{device}/?start={start.strftime("%Y-%m-%d %H:%M:%S.%f")}&stop={stop.strftime("%Y-%m-%d %H:%M:%S.%f")}&csv=false&concat_last=false"
    data = await return_data_drom_endpoint(endpoint)

    # Získání CO2 úrovní
    co2_levels = [entry['co2'] for device_data in data for entry in device_data['data'] if 'co2' in entry]

    # Pokud nejsou žádné hodnoty CO2, vrať prázdné počty
    if not co2_levels:
        return {"num_zero_crossings_up": 0, "num_zero_crossings_down": 0}

    # Vytvoření dataframe
    df = pd.DataFrame({'co2': co2_levels})

    # Výpočet rozdílů
    df['dy'] = df['co2'].diff()

    # Výpočet nulových průchodů
    zero_crossings_up = ((df['dy'] < 0) & (df['dy'].shift(-1) > 0)).sum()
    zero_crossings_down = ((df['dy'] > 0) & (df['dy'].shift(-1) < 0)).sum()

    zero_crossings_up = int(zero_crossings_up)
    zero_crossings_down = int(zero_crossings_down)
    return_data = {"num_zero_crossings_up": zero_crossings_up, "num_zero_crossings_down": zero_crossings_down}

    return JSONResponse(content=return_data)
@app.get("/records/average_durations/{place}", response_model=response_models.AverageDurationsResponseModel)
async def get_average_durations(
    place: str = Path(..., description="Url adresa místa", example="dcuk"),
    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00")):
    """
        Endpoint vrací průměrnou délku větrání a průměrnou délku zvyšování CO2 pro celou organizaci v daném časovém úseku.
        Výsledný čas je v minutách.
    """

    endpoint = f"{API_URL}/records/{place}/?start={start.strftime("%Y-%m-%d %H:%M:%S.%f")}&stop={stop.strftime("%Y-%m-%d %H:%M:%S.%f")}&concat_last=false&csv=false&children=false&only_key_devices=false"
    data = await return_data_drom_endpoint(endpoint)

    # Získání dat záznamů CO2 a časů
    records = [
        {"time": entry['time'], "co2": entry['co2']}
        for device_data in data
        for entry in device_data['data']
        if 'co2' in entry
    ]

    # Vrátí nulu pokud nejsou žádné hodnoty CO2
    if not records:
        return {"average_ventilation_time": 0, "average_increase_time": 0}

    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Diferenciál a časový rozdíl
    df['dy'] = df['co2'].diff()
    df['dt'] = df.index.to_series().diff().dt.total_seconds().fillna(0)

    # Přidání sloupce pro zjištění, zda se jedná o zvyšující se nebo klesající hodnoty
    df['is_increasing'] = df['dy'] > 0

    # Funkce na zjištění intervalů růstu a poklesu
    def get_intervals(df, direction):
        df['change'] = (df['is_increasing'] != df['is_increasing'].shift()).cumsum()
        interval_times = df[df['is_increasing'] == direction].groupby('change')['dt'].sum()
        return interval_times

    # Počítání, kdy hodnoty klesají (průměr)
    non_increasing_periods = get_intervals(df, direction=False)
    average_ventilation_time = round(non_increasing_periods.mean() / 60, 2)

    # Počítání, kdy hodnoty rostou (průměr)
    increasing_periods = get_intervals(df, direction=True)
    average_increase_time = round(increasing_periods.mean() / 60, 2)

    return JSONResponse(content={"average_ventilation_time": average_ventilation_time, "average_increase_time": average_increase_time})
@app.get("/records/analyze_co2_changes/place/{url}/", response_model=response_models.IncreaseDecreaseResponseModel)
async def analyze_place(
    url: str = Path(..., description="Url adresa místa", example="dcuk"),
    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00")
):

    endpoint = f"{API_URL}/records/{url}/?start={start}&stop={stop}&concat_last=false&csv=false&children=false&only_key_devices=false"

    data = await return_data_drom_endpoint(endpoint)

    co2_changes_up = 0
    co2_changes_down = 0

    for device in data:
        co2_values = [entry['co2'] for entry in device['data'] if 'co2' in entry]

        for i in range(1, len(co2_values)):
            deriv = co2_values[i] - co2_values[i - 1]
            if deriv > 0:
                co2_changes_up += 1
            elif deriv < 0:
                co2_changes_down += 1

    return JSONResponse(content={"co2_increase_count": co2_changes_up, "co2_decrease_count": co2_changes_down})
@app.get("/records/analyze_co2_dissipation/place/{url}/", response_model=response_models.CO2DissipationResponseModel)
async def analyze_place_co2_dissipation(
    url: str = Path(..., description="Url adresa místa", example="dcuk"),
    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00")
):
    """
    Vrací průměrný čas, který trvá, než se hodnota CO2 vrátí na normální úroveň (≤ 400 ppm) v daném časovém úseku.
    """
    endpoint = f"{API_URL}/records/{url}/?start={start.strftime("%Y-%m-%d %H:%M:%S.%f")}&stop={stop.strftime("%Y-%m-%d %H:%M:%S.%f")}&concat_last=false&csv=false&children=false&only_key_devices=false"

    data = await return_data_drom_endpoint(endpoint)

    dissipation_times = []
    peak_time = None

    for device in data:
        co2_data = [entry['co2'] for entry in device['data'] if 'co2' in entry]

        for i, co2_level in enumerate(co2_data):
            if co2_level > NORMAL_CO2_LEVEL and peak_time is None:
                peak_time = datetime.fromisoformat(device['data'][i]['time'])
            elif co2_level <= NORMAL_CO2_LEVEL and peak_time is not None:
                current_time = datetime.fromisoformat(device['data'][i]['time'])
                dissipation_time = (current_time - peak_time).total_seconds() / 60  # Převod na minuty
                dissipation_times.append(dissipation_time)
                peak_time = None

    if not dissipation_times:
        return {"average_dissipation_time_minutes": 0}

    average_dissipation_time = sum(dissipation_times) / len(dissipation_times)

    return JSONResponse(content={"average_dissipation_time_minutes": average_dissipation_time})
@app.get("/records/analyze_co2_dissipation/device/{device}/", response_model=response_models.CO2DissipationResponseModel)
async def analyze_co2_dissipation(
    device: str = Path(..., description="Název zařízení", example="eui-70b3d57ed006209f-co-05"),
    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00")
):
    """
    Vrací průměrný čas, který trvá, než se hodnota CO2 vrátí na normální úroveň (≤ 400 ppm) v daném časovém úseku.
    :param device:
    :param start:
    :param stop:
    :return:
    """
    endpoint = f"{API_URL}/records/place/{device}/?start={start.strftime("%Y-%m-%d %H:%M:%S.%f")}&stop={stop.strftime("%Y-%m-%d %H:%M:%S.%f")}&csv=false&concat_last=false"

    data = await return_data_drom_endpoint(endpoint)

    if not data or not isinstance(data, list) or not data[0].get('data'):
        raise HTTPException(status_code=500, detail="Invalid data format received from remote API")

    co2_data = data[0]['data']

    dissipation_times = []
    peak_time = None

    for entry in co2_data:
        co2_level = entry.get('co2')
        timestamp = entry.get('time')

        if co2_level is not None:
            if co2_level > NORMAL_CO2_LEVEL and peak_time is None:
                peak_time = datetime.fromisoformat(timestamp)
            elif co2_level <= NORMAL_CO2_LEVEL and peak_time is not None:
                current_time = datetime.fromisoformat(timestamp)
                dissipation_time = (current_time - peak_time).total_seconds() / 60  # Převod na minuty
                dissipation_times.append(dissipation_time)
                peak_time = None

    if not dissipation_times:
        return {"average_dissipation_time_minutes": 0}

    average_dissipation_time = sum(dissipation_times) / len(dissipation_times)

    return {"average_dissipation_time_minutes": average_dissipation_time}
#@app.get('/place/{url}/get_critical_sensors/', response_model=response_models.CriticalSensorsResponseModel)
#async def get_critical_sensors(
#    url: str = Path(..., description="Url adresa místa", example="dcuk"),
#    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
#    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00")
#):
#    """
#    Vrací čidla s kritickými hodnotami CO2 (≥ 1200) v daném časovém úseku.
#    """
#    try:
#        records = fetch_records(url, start.strftime("%Y-%m-%d %H:%M:%S.%f"), stop.strftime("%Y-%m-%d %H:%M:%S.%f"))
#
#        if not records:
#            raise HTTPException(status_code=404, detail="Žádné záznamy nebyly nalezeny")
#
#        critical_sensors = []
#
#        for record in records:
#            device_name = record["placement"] if record["placement"] else record["device"]
#            data_points = record.get("data", [])
#
#            for item in data_points:
#                co2 = item.get("co2")
#                if co2 is not None and co2 >= 1200:
#                    critical_sensors.append({"device": device_name, "co2": co2})
#
#        if not critical_sensors:
#            raise HTTPException(status_code=404, detail="Žádné kritické hodnoty CO2 nebyly nalezeny")
#
#        return JSONResponse(content={"critical_sensors": critical_sensors})
#
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_co2/", response_model=response_models.AnalyzeCO2ResponseModel)
async def analyze_co2(
    url: str = Query(..., description="Url adresa místa", example="dcuk"),
    start: datetime = Query(..., description="Časový údaj začátku ve formátu ISO 8601", example="2024-08-14T08:00:00"),
    stop: datetime = Query(..., description="Časový údaj konce ve formátu ISO 8601", example="2024-08-14T10:00:00"),
    co2_threshold: float = Query(..., description="Hodnota prahu CO2", example=1000),
    co2_rate_threshold: float = Query(..., description="Hodnota prahu rychlosti změny CO2", example=10),
):
    endpoint = f"{API_URL}/records/{url}/?start={start.strftime("%Y-%m-%d %H:%M:%S.%f")}&stop={stop.strftime("%Y-%m-%d %H:%M:%S.%f")}&concat_last=false&csv=false&children=false&only_key_devices=false"
    data = await return_data_drom_endpoint(endpoint)
    critical_devices = []

    for device in data:
        if not any('co2' in entry for entry in device['data']):
            continue

        times = np.array([np.datetime64(entry['time']) for entry in device['data']])
        co2_levels = np.array([entry['co2'] for entry in device['data']])

        time_diffs = np.diff(times).astype('timedelta64[s]').astype(int)

        if len(time_diffs) == 0:
            continue

        # Calculate the derivatives (rate of change)
        co2_derivative = np.diff(co2_levels) / time_diffs

        # Check if any value or its derivative exceeds the threshold
        if (np.any(co2_levels > co2_threshold) or
            np.any(np.abs(co2_derivative) > co2_rate_threshold)):
            critical_devices.append({"device": device['device']})

    if not critical_devices:
        raise HTTPException(status_code=200, detail="No critical devices found.")

    return JSONResponse(content={"critical_devices": critical_devices})

async def return_data_drom_endpoint(endpoint: str):
    '''
        Funkce sloužící pro vracení dat z endpointu z api
    '''
    async with httpx.AsyncClient() as client:
        response = await client.get(endpoint, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from remote API")
        data = response.json()

        return data
def fetch_records(url: str, start: str, stop: str) -> Dict:
    """
    Funkce pro stahování záznamů z API.
    """
    endpoint = f"/records/{url}/?start={start}&stop={stop}&concat_last=false&csv=false&children=false&only_key_devices=false"
    response = requests.get(API_URL + endpoint, headers=headers)
    return response.json()