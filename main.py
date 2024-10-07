import os
import requests
from datetime import datetime, time as dt_time, timedelta
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Union
from dotenv import load_dotenv
import httpx
import pandas as pd
from fastapi.responses import JSONResponse
import response_models
import numpy as np
from pykalman import KalmanFilter
import json

app = FastAPI()
NORMAL_CO2_LEVEL: int = 550

# Povolení CORS pro všechny zdroje
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/records/average_durations/", response_model=response_models.AverageDurationsResponse, summary="Získá průměrné časy ventilace a zvýšení CO2")
async def get_average_durations():
    """
    Tento endpoint vrací průměrný čas ventilace a průměrný čas zvýšení CO2
    pro celou organizaci v daném časovém období. Výsledky jsou vráceny v minutách.

    - `average_ventilation_time`: průměrná délka doby, kdy se CO2 snižuje (větrání)
    - `average_increase_time`: průměrná délka doby, kdy CO2 roste
    """
    try:
        # Čte data ze souboru week_data.json (týdenní data)
        data = read_records_from_json("./week_data.json")

        # Filtruje záznamy, kde hodnota CO2 je platná (CO2 > 0)
        records = [
            {"time": entry['time'], "co2": entry['co2']}
            for device_data in data
            for entry in device_data['data']
            if 'co2' in entry and entry['co2'] > 0  # Filtruje neplatné hodnoty CO2
        ]

        # Pokud nejsou k dispozici žádné validní záznamy, vrátí se 0 pro oba průměry
        if not records:
            return JSONResponse(content={"average_ventilation_time": 0, "average_increase_time": 0})

        # Vytvoření DataFrame z dat
        df = pd.DataFrame(records)
        df['time'] = pd.to_datetime(df['time'])  # Konverze časových údajů na datetime
        df.set_index('time', inplace=True)  # Nastavení času jako indexu

        # Výpočet změn CO2 a rozdílů v čase
        df['dy'] = df['co2'].diff().fillna(0)  # Změna hodnoty CO2 mezi záznamy
        df['dt'] = df.index.to_series().diff().dt.total_seconds().fillna(0)  # Rozdíl mezi časy v sekundách

        # Určení, zda hodnoty CO2 rostou (zvýšení) nebo klesají (větrání)
        df['is_increasing'] = df['dy'] > 0

        def get_intervals(df, direction: bool):
            """
            Vrací časová období, kdy CO2 buď roste, nebo klesá.
            - `direction=True`: rostoucí CO2
            - `direction=False`: klesající CO2 (větrání)
            """
            df['change'] = (df['is_increasing'] != df['is_increasing'].shift()).cumsum()  # Identifikace intervalů
            interval_times = df[df['is_increasing'] == direction].groupby('change')['dt'].sum()  # Součet časů pro každý interval
            return interval_times[interval_times > 0]  # Vrátí pouze platné intervaly

        # Získání intervalů, kdy CO2 klesá (větrání)
        non_increasing_periods = get_intervals(df, direction=False)
        average_ventilation_time = round(non_increasing_periods.mean() / 60, 2) if not non_increasing_periods.empty else 0

        # Získání intervalů, kdy CO2 roste
        increasing_periods = get_intervals(df, direction=True)
        average_increase_time = round(increasing_periods.mean() / 60, 2) if not increasing_periods.empty else 0

        # Odpověď ve formátu JSON
        return {"average_ventilation_time": average_ventilation_time, "average_increase_time": average_increase_time}

    except ValueError as e:
        # Pokud dojde k chybě ve vstupních datech, vrátí se chyba 400
        raise HTTPException(status_code=400, detail=f"Chyba hodnoty: {str(e)}")
    except Exception as e:
        # Pokud dojde k jiné chybě, vrátí se chyba 500
        raise HTTPException(status_code=500, detail=f"Neočekávaná chyba: {str(e)}")
@app.get("/records/analyze_co2_changes/place/", response_model=response_models.IncreaseDecreaseResponse, summary="Analyzuje změny CO2 na základě dat z určitého místa")
async def analyze_place():
    """
    Tento endpoint analyzuje data CO2 pro dané místo (z dat JSON) a vrací počet nárůstů a poklesů CO2.

    - `co2_increase_count`: počet nárůstů CO2 (při změně derivace z negativní na pozitivní)
    - `co2_decrease_count`: počet poklesů CO2 (při změně derivace z pozitivní na negativní)
    """
    try:
        # Čtení dat z JSON souboru
        data = read_records_from_json("./week_data.json")

        # Počáteční počty nárůstů a poklesů CO2
        co2_changes_up = 0
        co2_changes_down = 0

        # Procházení dat pro každé zařízení
        for device in data:
            # Získání hodnot CO2 pro jednotlivá zařízení
            co2_values = np.array([entry['co2'] for entry in device['data'] if 'co2' in entry])

            # Kontrola, zda máme alespoň 2 hodnoty pro výpočet derivace
            if len(co2_values) > 1:
                # Výpočet první derivace (změna CO2 mezi sousedními hodnotami)
                first_derivative = np.diff(co2_values)

                # Výpočet nulových průchodů pro nárůsty (kdy se derivace změní z negativní na pozitivní)
                zero_crossings_up = np.where(np.diff(np.sign(first_derivative)) > 0)[0]
                # Výpočet nulových průchodů pro poklesy (kdy se derivace změní z pozitivní na negativní)
                zero_crossings_down = np.where(np.diff(np.sign(first_derivative)) < 0)[0]

                # Aktualizace počtu nárůstů a poklesů CO2
                co2_changes_up += len(zero_crossings_up)
                co2_changes_down += len(zero_crossings_down)

        # Vrácení výsledků ve formátu JSON
        return JSONResponse(content={
            "co2_increase_count": co2_changes_up,  # Počet nárůstů CO2
            "co2_decrease_count": co2_changes_down  # Počet poklesů CO2
        })

    except Exception as e:
        # Pokud dojde k neočekávané chybě, vrátí se chyba 500
        raise HTTPException(status_code=500, detail=f"Neočekávaná chyba: {str(e)}")
@app.get("/records/analyze_co2_dissipation/place/", response_model=response_models.CO2DissipationResponseModel, summary="Analyzuje dobu návratu CO2 na normální hodnotu")
async def analyze_place_co2_dissipation():
    """
    Tento endpoint vrací průměrný čas, který trvá, než se hodnota CO2 vrátí na normální úroveň (≤ 400 ppm)
    v daném časovém úseku.

    - `average_dissipation_time_minutes`: průměrný čas návratu CO2 na normální úroveň (v minutách)
    """
    try:
        # Čtení dat z JSON souboru
        data = read_records_from_json("./week_data.json")

        # Seznam pro ukládání časů rozptylu CO2
        dissipation_times = []
        peak_time = None  # Čas, kdy byla zjištěna nejvyšší hodnota CO2

        # Procházení dat pro každé zařízení
        for device in data:
            # Získání hodnot CO2 pro každé zařízení
            co2_data = [entry['co2'] for entry in device['data'] if 'co2' in entry]

            # Procházení každé hodnoty CO2
            for i, co2_level in enumerate(co2_data):
                if co2_level > NORMAL_CO2_LEVEL and peak_time is None:
                    # Pokud CO2 překročí normální hladinu a nebyl ještě zaznamenán peak (špička)
                    peak_time = datetime.fromisoformat(device['data'][i]['time'])  # Zaznamenání času špičky
                elif co2_level <= NORMAL_CO2_LEVEL and peak_time is not None:
                    # Pokud se CO2 vrátí na normální hladinu a peak_time je nastavený
                    current_time = datetime.fromisoformat(device['data'][i]['time'])  # Čas návratu CO2 na normální úroveň
                    dissipation_time = (current_time - peak_time).total_seconds() / 60  # Výpočet rozptylového času v minutách
                    dissipation_times.append(dissipation_time)  # Uložení do seznamu
                    peak_time = None  # Reset peak_time pro další cyklus

        # Pokud nejsou žádné časy rozptylu, vrátí se 0
        if not dissipation_times:
            return {"average_dissipation_time_minutes": 0}

        # Výpočet průměrného času rozptylu CO2
        average_dissipation_time = round(sum(dissipation_times) / len(dissipation_times))

        # Odpověď ve formátu JSON
        return JSONResponse(content={"average_dissipation_time_minutes": average_dissipation_time})

    except Exception as e:
        # Pokud dojde k neočekávané chybě, vrátí se chyba 500
        raise HTTPException(status_code=500, detail=f"Neočekávaná chyba: {str(e)}")


@app.get("/records/predictions/", response_model=response_models.PredictionResponse,
         summary="Predikce trendů počtu lidí")
async def predict_people_trends():
    """
    Tento endpoint predikuje trendy počtu lidí na základě změn a akcelerace počtu lidí v objektu.
    Vrací významné časové úseky, ve kterých dochází k nárůstu nebo poklesu počtu lidí.

    - `data`: seznam časových úseků se změnou trendu v počtu lidí (nárůst/pokles)
    """
    try:
        # Získání odhadů počtu lidí z API
        people_estimation_response = await get_number_of_people_in_object()
        time_estimates = people_estimation_response.time_estimates

        # Pokud nejsou k dispozici žádné odhady, vrátí se prázdná data
        if not time_estimates:
            return JSONResponse(content={"data": []})

        # Převod odhadů na seznam slovníků pro další analýzu
        time_estimates_dicts = [{"time": estimate.time, "people_estimate": estimate.people_estimate}
                                for estimate in time_estimates]

        # Vytvoření DataFrame pro analýzu časových odhadů
        df = pd.DataFrame(time_estimates_dicts)
        df['time'] = pd.to_datetime(df['time'])  # Konverze časů na formát datetime
        df.set_index('time', inplace=True)  # Nastavení času jako indexu

        # Vyhlazení odhadu počtu lidí pomocí klouzavého průměru
        df['people_estimate_smoothed'] = df['people_estimate'].rolling(window=5, min_periods=1).mean()

        # Výpočet rozdílů a akcelerace (změna rozdílů)
        df['people_diff'] = df['people_estimate_smoothed'].diff()  # Změna v odhadu počtu lidí
        df['people_acceleration'] = df['people_diff'].diff()  # Změna změn (akcelerace)

        # Vyplnění případných chybějících hodnot nulou
        df.fillna(0, inplace=True)

        # Výpočet prahových hodnot pro akceleraci a změny na základě směrodatné odchylky
        acceleration_threshold = df['people_acceleration'].std() * 1.5
        diff_threshold = df['people_diff'].std() * 1.5

        # Identifikace významných změn (špiček) v akceleraci a změně počtu lidí
        significant_spikes = df[(df['people_acceleration'].abs() > acceleration_threshold) &
                                (df['people_diff'].abs() > diff_threshold)]

        # Určení, zda se jedná o nárůst nebo pokles počtu lidí
        significant_spikes['trend'] = np.where(significant_spikes['people_diff'] > 0, 'increase', 'decrease')

        # Sestavení seznamu časových úseků s významnými změnami trendu
        trend_periods = []
        last_end_time = None  # Poslední čas konce předchozího období

        # Procházení špiček a vytváření období s rozdílnými trendy
        for i in range(1, len(significant_spikes)):
            previous = significant_spikes.iloc[i - 1]
            current = significant_spikes.iloc[i]

            # Pokud dojde ke změně trendu (nárůst -> pokles nebo pokles -> nárůst)
            if previous['trend'] != current['trend']:
                start_time = previous.name.time()  # Začátek předchozího trendu
                end_time = current.name.time()  # Konec aktuálního trendu

                # Zajištění, že časové úseky nepřekrývají
                if last_end_time and start_time <= last_end_time:
                    start_time = (pd.Timestamp.combine(pd.Timestamp.now(), last_end_time) + pd.Timedelta(
                        minutes=1)).time()

                # Přidání časového úseku s trendem do výsledného seznamu
                trend_periods.append({
                    "trend": previous['trend'],  # Nárůst nebo pokles
                    "start_time": start_time.strftime("%H:%M:%S"),  # Začátek období
                    "end_time": end_time.strftime("%H:%M:%S")  # Konec období
                })
                last_end_time = end_time  # Aktualizace času konce

        # Pokud nebyly identifikovány žádné trendové období, vrátí se prázdná data
        if not trend_periods:
            return JSONResponse(content={"data": []})

        # Vrácení výsledků ve formátu JSON
        return JSONResponse(content={"data": trend_periods})

    except HTTPException as e:
        # Vrácení specifických chyb HTTP
        raise e
    except Exception as e:
        # Zachycení neočekávaných chyb a vrácení chyby 500
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze_co2/", response_model=response_models.AnalyzeCO2ResponseModel,
          responses={404: {"model": response_models.ErrorModelResponse}},
          summary="Analyzuje zařízení, která překračují prahové hodnoty CO2")
async def analyze_co2(
    co2_threshold: float = Query(..., description="Hodnota prahu CO2, nad kterou je zařízení považováno za kritické", example=1000),
    co2_rate_threshold: float = Query(..., description="Hodnota prahu rychlosti změny CO2, nad kterou je zařízení považováno za kritické", example=10),
):
    """
    Tento endpoint analyzuje data CO2 pro různá zařízení a vrací seznam zařízení, která překročila
    nastavené prahové hodnoty pro koncentraci CO2 nebo rychlost změny koncentrace CO2.

    - `co2_threshold`: prahová hodnota CO2 (ppm), nad kterou je zařízení považováno za kritické
    - `co2_rate_threshold`: prahová hodnota rychlosti změny CO2 (ppm/s), nad kterou je zařízení považováno za kritické
    - Pokud žádné zařízení nevyhovuje kritériím, vrátí se HTTP chyba 404.
    """
    try:
        # Čtení dat z JSON souboru
        data = read_records_from_json("./day_records.json")

        critical_devices = []  # Seznam kritických zařízení

        # Procházení dat pro každé zařízení
        for device in data:
            # Kontrola, zda jsou v datech hodnoty CO2
            if not any('co2' in entry for entry in device['data']):
                continue

            # Extrakce časů a hladin CO2
            times = np.array([np.datetime64(entry['time']) for entry in device['data']])
            co2_levels = np.array([entry['co2'] for entry in device['data']])

            # Výpočet rozdílů v čase (v sekundách) mezi záznamy
            time_diffs = np.diff(times).astype('timedelta64[s]').astype(int)

            # Pokud nejsou k dispozici žádné časové rozdíly, přeskočíme na další zařízení
            if len(time_diffs) == 0:
                continue

            # Výpočet derivace (změny) CO2 vzhledem k času
            co2_derivative = np.diff(co2_levels) / time_diffs

            # Kontrola, zda zařízení překračuje prahové hodnoty
            if (np.any(co2_levels > co2_threshold) or
                np.any(np.abs(co2_derivative) > co2_rate_threshold)):
                critical_devices.append({"device": device['device']})

        # Pokud nejsou nalezena žádná kritická zařízení, vrátíme HTTP chybu 404
        if not critical_devices:
            raise HTTPException(status_code=404, detail="No critical devices found.")

        # Vrácení seznamu kritických zařízení ve formátu JSON
        return JSONResponse(content={"critical_devices": critical_devices})

    except Exception as e:
        # Zachycení a vrácení nečekané chyby
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/records/number_of_people/place/", response_model=response_models.PeopleEstimationRM,
         responses={
             404: {"model": response_models.ErrorModelResponse},
             400: {"model": response_models.ErrorModelResponse},
             500: {"model": response_models.ErrorModelResponse}
         },
         summary="Odhad počtu lidí v objektu na základě CO2, vlhkosti a teploty")
async def get_number_of_people_in_object():
    """
    Tento endpoint analyzuje data CO2, vlhkosti a teploty a pomocí Kalmanova filtru odhaduje počet lidí v objektu.
    Vrací seznam odhadů počtu lidí pro různé časové úseky.

    - `time_estimates`: seznam časových odhadů počtu lidí
    - Pokud nejsou k dispozici žádná data, endpoint vrátí prázdný seznam.
    """
    try:
        # Načtení dat z JSON souboru
        records = read_records_from_json("./day_records.json")

        # Příprava seznamů pro uložení dat CO2, vlhkosti, teploty a časů
        co2_levels, humidity_levels, temperature_levels, times = [], [], [], []

        # Procházení zařízení a jejich dat
        for device in records:
            for data_point in device.get("data", []):
                # Získání datových bodů obsahujících CO2, vlhkost a teplotu
                if "co2" in data_point and "humidity" in data_point and "temperature" in data_point:
                    co2_levels.append(data_point["co2"])
                    humidity_levels.append(data_point["humidity"])
                    temperature_levels.append(data_point["temperature"])
                    times.append(datetime.fromisoformat(data_point["time"]))

        # Pokud nejsou k dispozici žádné hladiny CO2, vrátí se prázdná odpověď
        if not co2_levels:
            return response_models.PeopleEstimationRM(time_estimates=[])

        # Vytvoření DataFrame pro analýzu
        df = pd.DataFrame({'co2': co2_levels, 'humidity': humidity_levels, 'temperature': temperature_levels},
                          index=times)
        df.sort_index(inplace=True)  # Seřazení podle časového indexu

        # Určení nočního času, kdy je pravděpodobně méně lidí
        night_time = (df.index.time >= dt_time(22, 0)) | (df.index.time <= dt_time(7, 0))

        # Výpočet základní úrovně CO2 během nočního času
        baseline_co2 = df['co2'][night_time].median()

        # Pokud není k dispozici platná základní hodnota, použije se minimální hodnota CO2
        if np.isnan(baseline_co2):
            baseline_co2 = df['co2'].min()

        # Definování parametrů pro Kalmanův filtr
        transition_matrix = np.eye(3)
        observation_matrix = np.eye(3)
        initial_state_mean = [baseline_co2, np.mean(humidity_levels), np.mean(temperature_levels)]
        initial_state_covariance = np.eye(3)
        transition_covariance = np.eye(3) * 0.1
        observation_covariance = np.eye(3) * 1

        try:
            # Inicializace a aplikace Kalmanova filtru
            kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance
            )

            # Filtrování dat CO2, vlhkosti a teploty
            state_means, _ = kf.filter(df[['co2', 'humidity', 'temperature']].values)

        except Exception as e:
            # Zachycení chyb při aplikaci Kalmanova filtru
            raise HTTPException(status_code=500, detail=f"Error during Kalman filter application: {str(e)}")

        # Uložení odhadnutých hodnot CO2 po aplikaci Kalmanova filtru
        df['filtered_co2'] = state_means[:, 0]

        # Odhad počtu lidí na základě rozdílu mezi filtrovanými a základními hodnotami CO2
        df['people_estimate'] = (df['filtered_co2'] - baseline_co2) / 2.0

        # Odhad počtu lidí zaokrouhlený a omezený na minimum 0
        df['people_estimate'] = df['people_estimate'].apply(lambda x: max(int(round(x)), 0))

        # Vytvoření seznamu časových odhadů počtu lidí
        time_estimates = [{"time": time, "people_estimate": int(people)} for time, people in
                          zip(df.index, df['people_estimate'])]

        # Vrácení výsledků
        return response_models.PeopleEstimationRM(time_estimates=time_estimates)

    except ValueError as e:
        # Ošetření chyby hodnoty (např. špatná data)
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except HTTPException as e:
        # Znovuvyhození HTTP výjimky
        raise e
    except Exception as e:
        # Zachycení všech ostatních neočekávaných chyb
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/records/number_of_people_per_sensor/place/", response_model=List[response_models.DeviceData],
         summary="Odhad počtu lidí na základě senzorů CO2 pro jednotlivé senzory")
async def get_number_of_people_per_sensor():
    """
    Tento endpoint analyzuje data ze senzorů CO2 pro jednotlivé senzory a vrací odhad počtu lidí
    pro každé zařízení v konkrétních časových okamžicích. Odhady počtu lidí jsou založeny na rozdílu
    mezi hladinou CO2 a baseline hodnotou (získanou během nočního času).

    - `devices_data`: seznam zařízení a jejich odhadů počtu lidí v čase
    """
    try:
        # Načtení záznamů ze souboru JSON
        records = read_records_from_json("./day_records.json")

        # Pokud nejsou k dispozici žádná data, vrátíme chybu 404
        if records is None:
            raise HTTPException(status_code=404, detail="Data not found for the given URL.")

        devices_data = []  # Seznam zařízení pro uložení odhadů počtu lidí

        # Procházení každého zařízení v záznamech
        for record in records:
            co2_levels, times = [], []  # Seznamy pro uložení hodnot CO2 a časů

            # Získání hodnot CO2 a časových záznamů z dat zařízení
            for data_point in record["data"]:
                if "co2" in data_point:
                    co2_levels.append(data_point["co2"])
                    time = datetime.fromisoformat(data_point["time"])
                    times.append(time)

            # Pokud nejsou dostupné žádné hladiny CO2, přeskočíme na další zařízení
            if not co2_levels:
                continue

            # Vytvoření DataFrame s hodnotami CO2 a časovými záznamy
            df = pd.DataFrame({'co2': co2_levels}, index=times)

            # Seřazení dat podle časových záznamů
            df.sort_index(inplace=True)

            # Výpočet baseline hodnoty CO2 během nočního času (22:00 - 7:00)
            night_time = (df.index.time >= dt_time(22, 0)) | (df.index.time <= dt_time(7, 0))
            baseline_co2 = df['co2'][night_time].median()

            # Pokud baseline CO2 není dostupná, použije se minimální hodnota CO2
            if np.isnan(baseline_co2):
                baseline_co2 = df['co2'].min()

            # Definice parametrů pro Kalmanův filtr
            transition_matrix = np.eye(1)
            observation_matrix = np.eye(1)
            initial_state_mean = [baseline_co2]
            initial_state_covariance = np.eye(1)
            transition_covariance = np.eye(1) * 0.1
            observation_covariance = np.eye(1) * 1

            try:
                # Inicializace Kalmanova filtru
                kf = KalmanFilter(
                    transition_matrices=transition_matrix,
                    observation_matrices=observation_matrix,
                    initial_state_mean=initial_state_mean,
                    initial_state_covariance=initial_state_covariance,
                    transition_covariance=transition_covariance,
                    observation_covariance=observation_covariance
                )

                # Aplikace Kalmanova filtru na data CO2
                state_means, _ = kf.filter(df[['co2']].values)

            except Exception as e:
                # Zachycení chyb při aplikaci Kalmanova filtru a vrácení chyby 500
                raise HTTPException(status_code=500, detail=f"Error during Kalman filter application for device {record['device']}: {str(e)}")

            # Uložení filtrovaných hodnot CO2 a odhadu počtu lidí
            df['filtered_co2'] = state_means[:, 0]
            df['people_estimate'] = ((df['filtered_co2'] - baseline_co2) / 2.0).clip(0)

            # Vytvoření modelu dat pro každé zařízení a uložení výsledků
            device_data = response_models.DeviceData(
                device=record["device"],
                data=[
                    response_models.PeopleEstimationData(time=time.isoformat(), people_count=int(people))
                    for time, people in zip(df.index, df['people_estimate'])
                ]
            )

            # Přidání odhadů pro zařízení do seznamu
            devices_data.append(device_data)

        # Serializace výsledných dat zařízení
        serialized_devices_data = [device_data.dict() for device_data in devices_data]

        # Vrácení výsledků jako JSON odpověď
        return JSONResponse(content=serialized_devices_data)

    except ValueError as e:
        # Ošetření chyby hodnoty (např. špatná data) s odpovědí 400
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except Exception as e:
        # Ošetření neočekávané chyby s odpovědí 500
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/records/number_of_people/forecast/", response_model=response_models.PeopleEstimationsResponse,
         responses={
             404: {"model": response_models.PeopleEstimationsResponse},
             400: {"model": response_models.PeopleEstimationsResponse},
             500: {"model": response_models.PeopleEstimationsResponse}
         },
         summary="Predikce počtu lidí na základě senzorových dat s použitím Kalmanova filtru")
async def get_people_forecast(interval_minutes: int = Query(5, description="Časový interval v minutách pro resamplování dat")):
    """
    Tento endpoint provádí predikci počtu lidí v objektu na základě dat z CO2 senzorů,
    vlhkosti a teploty pomocí Kalmanova filtru. Data jsou průměrována podle zvoleného
    časového intervalu v minutách a následně se pro poslední dostupný den provádí predikce.

    - `interval_minutes`: časový interval v minutách pro průměrování dat
    - `time_estimates`: seznam časových odhadů počtu lidí, hladiny CO2 a času
    """
    try:
        # Načtení dat ze souboru JSON
        records = read_records_from_json("./week_data.json")

        # Příprava seznamů pro uložení dat CO2, vlhkosti, teploty a časů
        co2_levels, humidity_levels, temperature_levels, times = [], [], [], []

        # Procházení záznamů zařízení a jejich dat
        for device in records:
            for data_point in device.get("data", []):
                # Kontrola, zda záznam obsahuje CO2, vlhkost a teplotu
                if "co2" in data_point and "humidity" in data_point and "temperature" in data_point:
                    co2_levels.append(data_point["co2"])
                    humidity_levels.append(data_point["humidity"])
                    temperature_levels.append(data_point["temperature"])
                    times.append(datetime.fromisoformat(data_point["time"]))

        # Pokud nejsou k dispozici žádná data CO2, vrátí se prázdná odpověď
        if not co2_levels:
            return response_models.PeopleEstimationsResponse(time_estimates=[])

        # Vytvoření DataFrame pro analýzu dat
        df = pd.DataFrame({'co2': co2_levels, 'humidity': humidity_levels, 'temperature': temperature_levels}, index=times)

        # Resamplování (průměrování) dat podle zvoleného časového intervalu
        df_resampled = df.resample(f'{interval_minutes}min').mean()

        # Odstranění chybějících hodnot po resamplování
        df_resampled = df_resampled.dropna()

        # Definování Kalmanova filtru
        transition_matrix = np.eye(3)
        observation_matrix = np.eye(3)
        initial_state_mean = [df_resampled['co2'].iloc[0], df_resampled['humidity'].iloc[0], df_resampled['temperature'].iloc[0]]
        initial_state_covariance = np.eye(3)
        transition_covariance = np.eye(3) * 0.1
        observation_covariance = np.eye(3) * 1

        try:
            # Inicializace a aplikace Kalmanova filtru
            kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance
            )

            # Aplikace Kalmanova filtru na resamplovaná data
            state_means, _ = kf.filter(df_resampled[['co2', 'humidity', 'temperature']].values)

        except Exception as e:
            # Zachycení chyb při aplikaci Kalmanova filtru
            raise HTTPException(status_code=500, detail=f"Error during Kalman filter application: {str(e)}")

        # Uložení filtrovaných hodnot CO2
        df_resampled['filtered_co2'] = state_means[:, 0]

        # Odhad počtu lidí na základě filtrovaných dat CO2
        df_resampled['people_estimate'] = (df_resampled['filtered_co2'] - df_resampled['filtered_co2'].min()) / 2.0
        df_resampled['people_estimate'] = df_resampled['people_estimate'].apply(lambda x: max(int(round(x)), 0))

        # Úprava forecast date na poslední dostupný den v datech
        last_available_date = df_resampled.index.date[-1]
        forecast_date = last_available_date

        # Výběr dat pro poslední dostupný den
        df_resampled_forecast_day = df_resampled[df_resampled.index.date == forecast_date]

        # Pokud nejsou pro daný den k dispozici žádná data, vrátí se prázdná odpověď
        if df_resampled_forecast_day.empty:
            return response_models.PeopleEstimationsResponse(time_estimates=[])

        # Vytvoření seznamu časových odhadů počtu lidí
        time_estimates = [
            {"time": dt_time(t.hour, t.minute), "people_estimate": int(people), "co2_level": round(co2, 2)}
            for t, people, co2 in zip(df_resampled_forecast_day.index, df_resampled_forecast_day['people_estimate'], df_resampled_forecast_day['co2'])
        ]

        # Vrácení odpovědi s časovými odhady
        return response_models.PeopleEstimationsResponse(time_estimates=time_estimates)

    except ValueError as e:
        # Ošetření chyb spojených s nesprávnými hodnotami
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except HTTPException as e:
        # Znovuvyhození specifické HTTP výjimky
        raise e
    except Exception as e:
        # Zachycení neočekávaných chyb a vrácení chyby 500
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def read_records_from_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)