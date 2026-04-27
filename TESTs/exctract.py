#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:54:15 2026

@author: abhner
"""

import requests
import time

API_KEY = "caaddaa4f3de0daa0e043058306cf52d"
BASE_URL = "https://www.tng-project.org/api/TNG50-1/snapshots/99/subhalos"
SOLAR_Z = 0.0127

headers = {"api-key": API_KEY}

# ============================================
# FUNÇÃO PARA ACESSAR A API
# ============================================
def get_subhalo_data(subfind_id, fields=None, max_retries=3, sleep_time=0.5):
    """
    Busca os dados de um subhalo na API da TNG.
    
    Parameters
    ----------
    subfind_id : int
        ID do subhalo no snapshot 99.
    fields : list or None
        Lista de campos desejados. Se None, pega tudo.
    max_retries : int
        Número máximo de tentativas em caso de falha.
    sleep_time : float
        Pausa entre chamadas para não sobrecarregar a API.
    """
    url = f"{BASE_URL}/{int(subfind_id)}/"

    if fields is not None:
        url += "?" + "&".join([f"{field}" for field in fields])

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            time.sleep(sleep_time)
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"[Tentativa {attempt+1}/{max_retries}] Erro ao acessar ID {subfind_id}: {e}")
            time.sleep(1.0)

    return None

# ============================================
# GARANTIR A NOVA COLUNA
# ============================================
new_col = "logStarZ_99_75dex_new"

if new_col not in PaperII.columns:
    PaperII[new_col] = np.nan

# ============================================
# LOOP PRINCIPAL
# ============================================
for idx, row in PaperII.iterrows():
    subfind_id = row["SubfindID_99"]

    # testa se o valor original está faltando
    if pd.isna(row["logStarZ_99_75dex"]):
        data = get_subhalo_data(subfind_id)

        if data is None:
            print(f"Falha definitiva para ID {subfind_id}")
            continue

        starmetallicity = data.get("starmetallicity", np.nan)

        if pd.notna(starmetallicity) and starmetallicity > 0:
            new_value = np.log10(starmetallicity / SOLAR_Z) - 0.75
            PaperII.at[idx, new_col] = new_value
            print(f"ID {subfind_id}: starmetallicity={starmetallicity:.6e}, novo valor={new_value:.4f}")
        else:
            print(f"ID {subfind_id}: starmetallicity inválido ({starmetallicity})")
            
            
#%%¨

BadFlagAllSB_TEST['is_badflag_api'] = BadFlagAllSB_TEST['Flags'] == 0

print(BadFlagAllSB_TEST[['SubfindID_99', 'Flags', 'is_badflag_api']].head())
print(BadFlagAllSB_TEST['is_badflag_api'].value_counts(dropna=False))

#%%

import numpy as np
import pandas as pd
import requests
import time


def get_subhalo_flag(subfind_id, max_retries=3, sleep_time=0.3):
    url = f"{BASE_URL}/{int(subfind_id)}/"

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            time.sleep(sleep_time)
            return data.get("subhaloflag", np.nan)
        except requests.exceptions.RequestException as e:
            print(f"Erro no ID {subfind_id} (tentativa {attempt+1}/{max_retries}): {e}")
            time.sleep(1)

    return np.nan

# cria coluna nova
if "SubhaloFlag_API_99" not in BadFlagAllSB_TEST.columns:
    BadFlagAllSB_TEST["SubhaloFlag_API_99"] = np.nan

for idx, row in BadFlagAllSB_TEST.iterrows():
    subfind_id = row["SubfindID_99"]

    # só consulta se ainda não tiver valor salvo
    if pd.isna(row["SubhaloFlag_API_99"]):
        flag = get_subhalo_flag(subfind_id)
        BadFlagAllSB_TEST.at[idx, "SubhaloFlag_API_99"] = flag
        print(f"ID {subfind_id} -> SubhaloFlag = {flag}")

# classificação booleana
BadFlagAllSB_TEST["is_BadFlag"] = BadFlagAllSB_TEST["SubhaloFlag_API_99"] == 0
BadFlagAllSB_TEST["is_GoodFlag"] = BadFlagAllSB_TEST["SubhaloFlag_API_99"] == 1

print(BadFlagAllSB_TEST["SubhaloFlag_API_99"].value_counts(dropna=False))

#%%

not_bad = BadFlagAllSB_TEST.loc[BadFlagAllSB_TEST["SubhaloFlag_API_99"] != 0,
                                ["SubfindID_99", "SubhaloFlag_API_99"]]

print(not_bad)
print(f"Número que NÃO são BadFlag: {len(not_bad)}")