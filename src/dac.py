import numpy as np
from sequential import sequential_best_row


def dac_row_counts(U: np.ndarray) -> np.ndarray:
    """
    Divide & Conquer ile satır toplamlarını hesaplar.
    U: (R x T) 0/1 matrisi
    return: counts (R,) -> her satırdaki toplam 1 sayısı
    """
    R, T = U.shape
    if T == 1:
        return U[:, 0].astype(int)

    mid = T // 2
    left = U[:, :mid]      
    right = U[:, mid:]     
    left_counts = dac_row_counts(left)
    right_counts = dac_row_counts(right)
    return left_counts + right_counts


def _dac_argmax_range(counts: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """
    counts[start:end] aralığında D&C 'tournament' ile argmax bulur.
    Tie durumunda daha küçük index kazanır.
    """
    if end - start == 1:
        return start, int(counts[start])

    mid = (start + end) // 2
    left_idx, left_val = _dac_argmax_range(counts, start, mid)
    right_idx, right_val = _dac_argmax_range(counts, mid, end)

    # Tournament 
    if left_val > right_val:
        return left_idx, left_val
    elif right_val > left_val:
        return right_idx, right_val
    else:
        if left_idx < right_idx:
            return left_idx, left_val
        else:
            return right_idx, right_val


def dac_best_row(U: np.ndarray) -> tuple[int, int]:
    """
    1) dac_row_counts ile her satırdaki 1 sayısını bulur.
    2) Bu toplamlar üzerinde D&C argmax turnuvası yapar.
    Çıktı: (best_idx, best_val)
    """
    counts = dac_row_counts(U)
    R = counts.shape[0]

    best_idx, best_val = _dac_argmax_range(counts, 0, R)
    return best_idx, best_val


if __name__ == "__main__":

    U = np.loadtxt("../data/occupancy.csv", delimiter=",", dtype=int)

    if U.ndim == 1:
        U = U.reshape(1, -1)

    idx_seq, cnt_seq = sequential_best_row(U)
    print(f"Sequential → best row = {idx_seq}, ones = {cnt_seq}")

    idx_dac, cnt_dac = dac_best_row(U)
    print(f"D&C        → best row = {idx_dac}, ones = {cnt_dac}")
    
    if (idx_seq, cnt_seq) != (idx_dac, cnt_dac):
        print("⚠ WARNING: Sequential ve D&C sonuçları farklı!")
    else:
        print("✅ OK: D&C ve Sequential aynı sonucu verdi.")

