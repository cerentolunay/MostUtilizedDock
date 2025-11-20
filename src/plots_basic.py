import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from datetime import datetime, timedelta

# occupancy.csv'deki verileri kulalnarak Heatmap ve Bar Chart grafiklerini üretir.

ROOT_DIR =Path(__file__).resolve().parent.parent
DATA_DIR=ROOT_DIR/"data"
PLOTS_DIR=ROOT_DIR/"plots"
PLOTS_DIR.mkdir(exist_ok=True)

def save_heatmap(U: np.ndarray,path: str | Path = PLOTS_DIR / "heatmap.png",delta_minutes: int = 5,day_start_str: str = "06:30") -> None:
    """
    Verilen doluluk matrisini (U) kullanarak bir Heatmap oluşturur ve kaydeder.
    
        U (np.ndarray): R x T boyutunda doluluk matrisi (0 ve 1'lerden oluşur).
        path (str | Path): Grafiğin kaydedileceği dosya yolu.
        delta_minutes (int): Her bir zaman diliminin (slot) kaç dakika olduğu.
        day_start_str (str): Günün başlangıç saati (format: "HH:MM").
    """
    
    path = Path(path)
    R, T = U.shape

    start_h, start_m = map(int, day_start_str.split(":"))
    base_time = datetime(2025, 1, 1, start_h, start_m)
    slot_delta = timedelta(minutes=delta_minutes)
    step = 12
    tick_positions = list(range(0, T, step))
    tick_labels = []

    for i in tick_positions:
        t0 = base_time + i * slot_delta
        tick_labels.append(t0.strftime("%H:%M"))

    plt.figure(figsize=(12, 4)) 
    plt.imshow(U, aspect="auto", cmap="spring", interpolation="nearest")
    plt.yticks(np.arange(R), [f"Dock-{i+1}" for i in range(R)])
    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")
    plt.grid(which="both", color="white", linewidth=0.3)
    plt.xlabel("Time Slot")
    plt.ylabel("Dock")
    plt.title("Dock Occupancy Matrix (rows = docks, columns = time slots; 1 = occupied)")
    cbar = plt.colorbar()
    cbar.set_label("Occupancy")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path, dpi=200)
    plt.close()
     

def save_bars(U:np.ndarray, path:str | Path=PLOTS_DIR/"bar_chart.png")-> None:
    
    # Her bir iskelenin (dock) toplam dolu olduğu slot sayısını hesaplar ve Bar Chart çizer.
 
    path=Path(path)
    totals=U.sum(axis=1)
    R=U.shape[0]
    x=np.arange(R)

    plt.figure(figsize=(10, 5))
    plt.bar(x, totals, color="#F113FD")  
    plt.xticks(x, [f"Dock-{i+1}" for i in range(R)], rotation=0)
    plt.xlabel("Dock")
    plt.ylabel("Occupied Slots Count")
    plt.title("Total Occupied Slots per Dock")

    for i, v in enumerate(totals):
        plt.text(i, v + 2, str(int(v)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    """
    1. occupancy.csv dosyasını yükler.
    2. Heatmap ve Bar Chart fonksiyonlarını çağırır.
    """
    occ_path=DATA_DIR/"occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(f"occupancy.csv bulunamadı: {occ_path}")
    U=np.loadtxt(occ_path, delimiter =",",dtype=int)
    print(f"U shape: {U.shape}")
    print("Heatmap ve bar chart üretiliyor...")
    save_heatmap(U)
    save_bars(U)
    print(f"✓ Heatmap:    {PLOTS_DIR / 'heatmap.png'}")
    print(f"✓ Bar chart:  {PLOTS_DIR / 'bar_chart.png'}")
  
if __name__ == "__main__":
    main()
