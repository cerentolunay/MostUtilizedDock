import numpy as np
import pandas as pd
from datetime import timedelta
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#C:/Users/VICTUS/MostUtilizedDock/data

def build_time_grid(day_start,day_end,delta_minutes):
    slots = []
    current = day_start
    delta = timedelta(minutes=delta_minutes)
    while current <day_end:
        slots.append(current)
        current+= delta
    return slots

def summarize_matrix(U, delta_minutes):
    ones= int(U.sum())
    R,T = U.shape
    sparsity = 1-ones/(R*T)
    return{
        "R": int(R),
        "T": int(T),
        "ones": ones,
        "sparsity": sparsity,
        "delta_minutes": delta_minutes,
    }

def main():
    csv_path = DATA_DIR/ "events.csv"
    delta_minutes = 5
    df= pd.read_csv(csv_path)
    df=df.rename(
        columns={
            "dock_id": "dock_id",
            "arrival_time": "arrival_time",
            "departure_time": "departure_time",
        }
    )
    df["tin"]=pd.to_datetime(df["arrival_time"])
    df["tout"]=pd.to_datetime(df["departure_time"])
    day_start = df["tin"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + pd.Timedelta(days=1)

    slots = build_time_grid(day_start, day_end, delta_minutes)
    T = len(slots)
    
    docks = sorted(df["dock_id"].unique().tolist())
    R = len(docks)

    print(f"Gün: {day_start.date()}, R={R}, T={T}, delta={delta_minutes}dk")
#
    U= np.zeros((R,T),dtype=int)    
    dock_to_index={dock_id: idx for idx, dock_id in enumerate(docks)}

    for _,row in df.iterrows():
        dock_id=row["dock_id"]
        tin=row["tin"]
        tout=row["tout"]
        i=dock_to_index[dock_id]
        for t_idx, slot_start in enumerate(slots):
            slot_end=slot_start + pd.Timedelta(minutes=delta_minutes)
            if slot_end<= tin or slot_start>=tout:
                continue
            U[i, t_idx]=1

    occupancy_path=DATA_DIR/"occupancy.csv"
    np.savetxt(occupancy_path,U,fmt="%d",delimiter=",")
    summary=summarize_matrix(U,delta_minutes)
    info_path = DATA_DIR / "info.json"

    with info_path.open("w",encoding="utf-8") as f:
        json.dump(summary,f,indent=4)
    print("U matrisi occupancy.csv olarak kaydedildi.")
    print("Özet info.json olarak kaydedildi.")
    print("Özet:", summary)

if __name__ == "__main__":
    main()