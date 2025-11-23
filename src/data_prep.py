import numpy as np
import pandas as pd
from datetime import timedelta
import json
from pathlib import Path

# data/ klasörünün kökünü ayarla
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_time_grid(day_start, day_end, delta_minutes: int) -> list[pd.Timestamp]:
    """
    Verilen gün aralığını, delta_minutes dakikalık slotlara böler.

    Örneğin:
        day_start = 00:00, day_end = 24:00, delta_minutes = 5
    ise 5 dakikalık ardışık zaman damgalarından oluşan bir liste döner.
    """
    slots: list[pd.Timestamp] = []
    current = day_start
    delta = timedelta(minutes=delta_minutes)

    while current < day_end:
        slots.append(current)
        current += delta

    return slots


def summarize_matrix(U: np.ndarray, delta_minutes: int) -> dict:
    """
    U matrisi için özet istatistikleri üretir.

    Dönen sözlük:
        R             → satır sayısı (dock sayısı)
        T             → sütun sayısı (time slot sayısı)
        ones          → matristeki toplam 1 sayısı
        sparsity      → 1 - (ones / (R * T))
        delta_minutes → slot uzunluğu (dakika)
    """
    ones = int(U.sum())
    R, T = U.shape
    sparsity = 1 - ones / (R * T)

    return {
        "R": int(R),
        "T": int(T),
        "ones": ones,
        "sparsity": sparsity,
        "delta_minutes": delta_minutes,
    }


def main() -> None:
    """
    1. events.csv dosyasını okur.
    2. Gün için time slot grid'ini oluşturur.
    3. Her dock ve slot için occupancy matrisi U'yu doldurur.
       - Kural: slot ile [tin, tout) aralığı arasında herhangi bir çakışma varsa U[i, t] = 1
    4. U'yu occupancy.csv olarak kaydeder.
    5. Özet bilgileri info.json olarak kaydeder.
    """
    csv_path = DATA_DIR / "events.csv"
    delta_minutes = 5

    # Etkinlik verisini yükle (dock_id, arrival_time, departure_time)
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "dock_id": "dock_id",
            "arrival_time": "arrival_time",
            "departure_time": "departure_time",
        }
    )

    # Zaman sütunlarını datetime'a çevir
    df["tin"] = pd.to_datetime(df["arrival_time"])
    df["tout"] = pd.to_datetime(df["departure_time"])

    # Seçilen gün: ilk arrival_time'ın günü, 00:00 - 24:00 aralığı
    day_start = df["tin"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + pd.Timedelta(days=1)

    # Time slot listesini oluştur
    slots = build_time_grid(day_start, day_end, delta_minutes)
    T = len(slots)

    # Dock ID'lerini indekslere eşleştir
    docks = sorted(df["dock_id"].unique().tolist())
    R = len(docks)

    print(f"Gün: {day_start.date()}, R={R}, T={T}, delta={delta_minutes} dk")

    # U matrisi: başlangıçta her şey 0 (boş)
    U = np.zeros((R, T), dtype=int)
    dock_to_index = {dock_id: idx for idx, dock_id in enumerate(docks)}

    # Her event için ilgili dock ve time slot'ları işaretle
    for _, row in df.iterrows():
        dock_id = row["dock_id"]
        tin = row["tin"]
        tout = row["tout"]
        i = dock_to_index[dock_id]

        # Her slot için "herhangi bir pozitif overlap var mı?" kontrolü
        for t_idx, slot_start in enumerate(slots):
            slot_end = slot_start + pd.Timedelta(minutes=delta_minutes)

            # Hiç çakışma yoksa devam et (disjoint interval)
            if slot_end <= tin or slot_start >= tout:
                continue

            # En az bir miktar çakışma varsa bu slot dolu kabul edilir
            U[i, t_idx] = 1

    # U matrisini CSV olarak kaydet
    occupancy_path = DATA_DIR / "occupancy.csv"
    np.savetxt(occupancy_path, U, fmt="%d", delimiter=",")

    # Özet istatistikleri hesapla ve info.json'a yaz
    summary = summarize_matrix(U, delta_minutes)
    info_path = DATA_DIR / "info.json"

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("U matrisi occupancy.csv olarak kaydedildi.")
    print("Özet info.json olarak kaydedildi.")
    print("Özet:", summary)


if __name__ == "__main__":
    main()
