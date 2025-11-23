import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from datetime import datetime, timedelta

# Bu dosya occupancy.csv'yi kullanarak iki grafik üretir:
# 1) Dock-Time heatmap
# 2) Dock başına toplam doluluk (bar chart)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def save_heatmap(
    U: np.ndarray,
    path: str | Path = PLOTS_DIR / "heatmap.png",
    delta_minutes: int = 5,
    day_start_str: str = "06:30"
) -> None:
    """
    Doluluk matrisi (U) kullanılarak heatmap görüntüsü oluşturur.

    Parametreler
    ------------
    U : np.ndarray
        R x T boyutunda, 0/1 değerlerinden oluşan doluluk matrisi.
    path : str | Path
        Kaydedilecek heatmap dosyasının yolu.
    delta_minutes : int
        Her bir zaman slotunun uzunluğu (dk).
    day_start_str : str
        Günün başlangıç zamanı ("HH:MM" formatında).
    """

    path = Path(path)
    R, T = U.shape

    # X-axis üzerindeki zaman etiketlerini oluşturmak için
    start_h, start_m = map(int, day_start_str.split(":"))
    base_time = datetime(2025, 1, 1, start_h, start_m)
    slot_delta = timedelta(minutes=delta_minutes)

    # X-tick'leri seyrek göstererek okunabilirliği arttırıyoruz
    step = 12   # 12*5 dk = 1 saatlik adım
    tick_positions = list(range(0, T, step))
    tick_labels = []

    for i in tick_positions:
        t0 = base_time + i * slot_delta
        tick_labels.append(t0.strftime("%H:%M"))

    # Heatmap çizimi
    plt.figure(figsize=(12, 4))
    plt.imshow(U, aspect="auto", cmap="spring", interpolation="nearest")

    plt.yticks(np.arange(R), [f"Dock-{i+1}" for i in range(R)])
    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

    plt.grid(which="both", color="white", linewidth=0.3)
    plt.xlabel("Time")
    plt.ylabel("Dock")
    plt.title("Dock Occupancy Matrix (1 = occupied)")

    cbar = plt.colorbar()
    cbar.set_label("Occupancy")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path, dpi=200)
    plt.close()


def save_bars(U: np.ndarray, path: str | Path = PLOTS_DIR / "bar_chart.png") -> None:
    """
    Dock başına toplam dolu slot sayısını hesaplayıp bar chart olarak kaydeder.

    Parametreler
    ------------
    U : np.ndarray
        R x T doluluk matrisi.
    path : str | Path
        Kaydedilecek grafik yolu.
    """

    path = Path(path)

    # Her satırın toplam doluluk sayısını hesapla
    totals = U.sum(axis=1)
    R = U.shape[0]
    x = np.arange(R)

    plt.figure(figsize=(10, 5))
    plt.bar(x, totals, color="#F113FD")

    plt.xticks(x, [f"Dock-{i+1}" for i in range(R)], rotation=0)
    plt.xlabel("Dock")
    plt.ylabel("Occupied Slots Count")
    plt.title("Total Occupied Slots per Dock")

    # Barların üstüne değer yaz
    for i, v in enumerate(totals):
        plt.text(i, v + 2, str(int(v)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    """
    occupancy.csv dosyasını yükler ve hem heatmap hem bar chart üretir.
    """

    occ_path = DATA_DIR / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(f"occupancy.csv bulunamadı: {occ_path}")

    U = np.loadtxt(occ_path, delimiter=",", dtype=int)

    print(f"U shape: {U.shape}")
    print("Heatmap ve bar chart üretiliyor...")

    save_heatmap(U)
    save_bars(U)

    print(f"✓ Heatmap kaydedildi:   {PLOTS_DIR / 'heatmap.png'}")
    print(f"✓ Bar chart kaydedildi: {PLOTS_DIR / 'bar_chart.png'}")


if __name__ == "__main__":
    main()
