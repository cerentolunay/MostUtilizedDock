import numpy as np
from sequential import sequential_best_row  

def dac_row_counts(U:np.ndarray) -> np.ndarray:
    R,T= U.shape
    if T==1:
        return U[:,0].astype(int)
    mid=T//2
    left =U[:,:mid] # sol yarı (R x mid)
    right=U[:,mid:]  # sağ yarı (R x (T - mid))
    left_counts=dac_row_counts(left)
    right_counts=dac_row_counts(right)
    return left_counts+right_counts

def dac_best_row(U:np.ndarray) -> tuple[int,int]:
    counts=dac_row_counts(U)
    R=counts.shape[0]
    best_idx=0
    best_val=int(counts[0])
    for i in range(1,R):
        c=int(counts[i])
        if c>best_val:
            best_val=c
            best_idx=i

    return best_idx,best_val

if __name__ == "__main__":
    # occupancy.csv'den U matrisini oku
    U = np.loadtxt("../data/occupancy.csv", delimiter=",", dtype=int)

    # Sequential sonuç
    idx_seq, cnt_seq = sequential_best_row(U)
    print(f"Sequential → best row = {idx_seq}, ones = {cnt_seq}")

    # D&C sonuç
    idx_dac, cnt_dac = dac_best_row(U)
    print(f"D&C        → best row = {idx_dac}, ones = {cnt_dac}")

    # Güvenlik: ikisi de aynı sonucu vermeli
    if (idx_seq, cnt_seq) != (idx_dac, cnt_dac):
        print("⚠ WARNING: Sequential ve D&C sonuçları farklı!")
    else:
        print("✅ OK: D&C ve Sequential aynı sonucu verdi.")      