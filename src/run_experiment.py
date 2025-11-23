import time
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sequential import sequential_best_row
from dac import dac_best_row

# Ana klasör yapısı
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TIMINGS_CSV = RESULTS_DIR / "timings.csv"


def time_method(func, U: np.ndarray, n_runs: int = 20) -> tuple[float, float, list[float]]:
    """
    Verilen fonksiyonu (sequential veya D&C) aynı U matrisi üzerinde
    n_runs kez çalıştırıp çalışma sürelerini ölçer.

    Dönüş:
        mean → ortalama süre
        std  → standart sapma
        times → her tekrarın tekil çalışma süreleri
    """

    times: list[float] = []

    # Küçük bir warm-up: cache, import, ilk çağrı overhead'ini azaltır
    func(U)

    for _ in range(n_runs):
        t0 = time.perf_counter()      # yüksek çözünürlüklü zaman ölçer
        func(U)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std, times


def append_timings_csv(method: str, R: int, T: int, times: list[float], csv_path: Path = TIMINGS_CSV) -> None:
    """
    Her (method, R, T) kombinasyonu için tüm tekil çalışmaları
    results/timings.csv dosyasına yazar.

    CSV formatı:
        method,R,T,repeat,seconds
    """
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # İlk kez yazılıyorsa header ekle
        if not file_exists:
            writer.writerow(["method", "R", "T", "repeat", "seconds"])

        for rep_idx, sec in enumerate(times, start=1):
            writer.writerow([method, R, T, rep_idx, f"{sec:.8e}"])


def run_correctness_check(U: np.ndarray) -> dict:
    """
    Sequential ve D&C algoritmalarının aynı sonucu üretip üretmediğini test eder.
    """

    idx_seq, cnt_seq = sequential_best_row(U)
    idx_dac, cnt_dac = dac_best_row(U)

    equal = (idx_seq == idx_dac) and (cnt_seq == cnt_dac)

    print("=== Correctness Check ===")
    print(f"Sequential → best row = {idx_seq}, ones = {cnt_seq}")
    print(f"D&C        → best row = {idx_dac}, ones = {cnt_dac}")
    if equal:
        print("✓ OK: Sequential ve D&C aynı sonucu verdi.\n")
    else:
        print("⚠ WARNING: FARKLI SONUÇ ÜRETİLDİ!\n")

    return {
        "sequential": {"best_row": int(idx_seq), "ones": int(cnt_seq)},
        "dac": {"best_row": int(idx_dac), "ones": int(cnt_dac)},
        "equal": bool(equal),
    }


def run_full_matrix_timing(U: np.ndarray, n_runs: int = 20) -> dict:
    """
    Tam U matrisi üzerinde sequential ve D&C algoritmalarının runtime ölçümünü yapar.
    Sonuçları CSV'ye kaydeder.
    """
    R, T = U.shape
    print("=== Full Matrix Timing ===")
    print(f"U shape: R={R}, T={T}, runs={n_runs}")

    seq_mean, seq_std, seq_times = time_method(sequential_best_row, U, n_runs=n_runs)
    dac_mean, dac_std, dac_times = time_method(dac_best_row, U, n_runs=n_runs)

    append_timings_csv("sequential_full", R, T, seq_times)
    append_timings_csv("dac_full", R, T, dac_times)

    print(f"Sequential → mean = {seq_mean:.6f}s, std = {seq_std:.6f}s")
    print(f"D&C        → mean = {dac_mean:.6f}s, std = {dac_std:.6f}s\n")

    return {
        "n_runs": n_runs,
        "sequential": {"mean": seq_mean, "std": seq_std},
        "dac": {"mean": dac_mean, "std": dac_std},
    }


def run_scale_experiment(U: np.ndarray, n_runs: int = 10) -> dict:
    """
    T (sütun sayısı) farklılaştırılarak (48, 96, 144 vs.)
    her iki metodun çalışma sürelerini ölçer ve runtime-vs-T grafiğini üretir.
    """
    R, T_full = U.shape

    # T değerlerini otomatik oluştur: 6 eşit parça
    num_points = 6
    T_values = sorted(set(int(T_full * k / num_points) for k in range(1, num_points + 1)))
    T_values = [T for T in T_values if T > 0]

    print("=== Runtime vs T (Scale Experiment) ===")
    print(f"R sabit = {R}, T değerleri: {T_values}")
    print(f"Her T için {n_runs} tekrar yapılıyor.\n")

    seq_means = []
    dac_means = []
    seq_stds = []
    dac_stds = []

    per_T_results: dict[str, dict] = {}

    for T in T_values:
        U_sub = U[:, :T]  # sadece ilk T sütun

        print(f"-- T = {T} --")
        seq_mean, seq_std, seq_times = time_method(sequential_best_row, U_sub, n_runs=n_runs)
        dac_mean, dac_std, dac_times = time_method(dac_best_row, U_sub, n_runs=n_runs)

        print(f"Sequential → mean = {seq_mean:.6f}s, std = {seq_std:.6f}s")
        print(f"D&C        → mean = {dac_mean:.6f}s, std = {dac_std:.6f}s\n")

        append_timings_csv("sequential_T", R, T, seq_times)
        append_timings_csv("dac_T", R, T, dac_times)

        seq_means.append(seq_mean)
        dac_means.append(dac_mean)
        seq_stds.append(seq_std)
        dac_stds.append(dac_std)

        per_T_results[str(T)] = {
            "sequential": {"mean": seq_mean, "std": seq_std},
            "dac": {"mean": dac_mean, "std": dac_std},
        }

    # Grafik oluşturma
    plt.figure(figsize=(8, 5))
    plt.plot(T_values, seq_means, marker="o", label="Sequential")
    plt.plot(T_values, dac_means, marker="o", label="Divide & Conquer")
    plt.xlabel("Number of time slots (T)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs T for Sequential and D&C")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = PLOTS_DIR / "runtime_vs_T.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"✓ Runtime vs T plot kaydedildi: {plot_path}\n")

    return {
        "T_values": T_values,
        "sequential_means": seq_means,
        "sequential_stds": seq_stds,
        "dac_means": dac_means,
        "dac_stds": dac_stds,
        "per_T": per_T_results,
    }


def main():
    # Önceki ölçüm CSV’si varsa sil
    if TIMINGS_CSV.exists():
        TIMINGS_CSV.unlink()

    # Occupancy matrix'i yükle
    occ_path = DATA_DIR / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(f"occupancy.csv bulunamadı: {occ_path}")

    U = np.loadtxt(occ_path, delimiter=",", dtype=int)
    R, T = U.shape
    print(f"=== Experiment on U (R={R}, T={T}) ===\n")

    # 1) Doğruluk testi
    correctness = run_correctness_check(U)

    # 2) Tam matris runtime testi
    full_timing = run_full_matrix_timing(U, n_runs=20)

    # 3) Farklı T değerlerinde ölçekleme deneyi
    scale_results = run_scale_experiment(U, n_runs=10)

    # 4) Sonuçları JSON'a kaydet
    results = {
        "U_shape": {"R": int(R), "T": int(T)},
        "correctness": correctness,
        "full_matrix_timing": full_timing,
        "scale_experiment": scale_results,
    }

    json_path = RESULTS_DIR / "times.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✓ JSON sonuç kaydedildi → {json_path}")
    print(f"✓ Tekil ölçümler CSV kaydedildi → {TIMINGS_CSV}")


if __name__ == "__main__":
    main()
