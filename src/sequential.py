import numpy as np

def sequential_best_row(U: np.ndarray) -> tuple[int, int]:
    # Simple baseline algorithm:
    # For each row, count how many slots are occupied (sum of 1's).
    # Keep the row with the highest count. Ties are resolved naturally
    # because we only update when a strictly larger count is found.

    R, T = U.shape
    best_idx = 0
    best_count = 0

    for i in range(R):
        cnt = int(U[i].sum())   # number of occupied slots in row i
        if cnt > best_count:
            best_count = cnt
            best_idx = i

    return best_idx, best_count


if __name__ == "__main__":
    # Load binary occupancy matrix
    U = np.loadtxt("../data/occupancy.csv", delimiter=",", dtype=int)

    # Run the sequential algorithm
    idx, cnt = sequential_best_row(U)
    print(f"Sequential → best row = {idx}, ones = {cnt}")
