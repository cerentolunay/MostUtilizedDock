import numpy as np
def sequential_best_row(U:np.ndarray)->tuple[int,int]:
    R,T=U.shape
    best_idx=0
    best_count=0
    for i in range(R):
        cnt=int(U[i].sum())
        if cnt>best_count:
            best_count = cnt
            best_idx = i
    return best_idx,best_count        

if __name__ == "__main__":
    U = np.loadtxt("../data/occupancy.csv", delimiter=",", dtype=int)
    idx, cnt = sequential_best_row(U)
    print(f"Sequential → best row = {idx}, ones = {cnt}")