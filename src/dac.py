import numpy as np
from sequential import sequential_best_row


def dac_row_counts(U: np.ndarray) -> np.ndarray:
    """
    Row-sum computation using Divide & Conquer.
    Matrisi (R x T) olacak şekilde sütunlardan ikiye bölerek
    her yarının satır toplamlarını hesaplar ve toplar.

    Parametre:
        U : np.ndarray (R x T)

    Dönen:
        counts : np.ndarray (R,)
            Her satırın toplam 1 sayısı
    """

    R, T = U.shape

    # Base case: Tek sütun varsa doğrudan geri dön
    if T == 1:
        return U[:, 0].astype(int)

    # Recursive split: Sol yarı / Sağ yarı
    mid = T // 2
    left = U[:, :mid]
    right = U[:, mid:]

    # Her iki Yarının row-sum sonuçlarını al
    left_counts = dac_row_counts(left)
    right_counts = dac_row_counts(right)

    # Combine: Vektörel toplama
    return left_counts + right_counts


def _dac_argmax_range(counts: np.ndarray, start: int, end: int) -> tuple[int, int]:
    """
    Tournament-style D&C argmax.
    Belirli bir [start, end) aralığında en büyük değeri bulur.
    Tie durumunda küçük indeks kazanır (assignment gereği).

    Dönüş:
        (best_idx, best_val)
    """

    # Base case: yalnızca tek element varsa
    if end - start == 1:
        return start, int(counts[start])

    # Aralığı ikiye böl
    mid = (start + end) // 2
    left_idx, left_val = _dac_argmax_range(counts, start, mid)
    right_idx, right_val = _dac_argmax_range(counts, mid, end)

    # Tournament logic
    if left_val > right_val:
        return left_idx, left_val
    elif right_val > left_val:
        return right_idx, right_val
    else:
        # Tie: küçük indeks kazanır
        return (left_idx, left_val) if left_idx < right_idx else (right_idx, right_val)


def dac_best_row(U: np.ndarray) -> tuple[int, int]:
    """
    D&C yaklaşımının tam versiyonu:
    1) Row-sum'ları D&C ile hesaplar.
    2) Tournament-style argmax ile en iyi satırı bulur.

    Dönen:
        (best_idx, best_val)
    """

    counts = dac_row_counts(U)
    R = counts.shape[0]

    best_idx, best_val = _dac_argmax_range(counts, 0, R)
    return best_idx, best_val


if __name__ == "__main__":
    # Hızlı test: Sequential ve D&C aynı sonucu veriyor mu?
    U = np.loadtxt("../data/occupancy.csv", delimiter=",", dtype=int)

    # Tek satırlı dosya durumuna karşı reshape
    if U.ndim == 1:
        U = U.reshape(1, -1)

    # Sequential sonuç
    idx_seq, cnt_seq = sequential_best_row(U)
    print(f"Sequential → best row = {idx_seq}, ones = {cnt_seq}")

    # D&C sonuç
    idx_dac, cnt_dac = dac_best_row(U)
    print(f"D&C        → best row = {idx_dac}, ones = {cnt_dac}")

    # Doğruluk kontrolü
    if (idx_seq, cnt_seq) != (idx_dac, cnt_dac):
        print("⚠ WARNING: Sequential ve D&C sonuçları FARKLI!")
    else:
        print("✓ OK: D&C ve Sequential aynı sonucu verdi.")
