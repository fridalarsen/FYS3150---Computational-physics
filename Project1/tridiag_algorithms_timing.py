import numpy as np
import time
from tridiag_matrix_algorithm import run_tma
from special_tridiag_matrix_algorithm import special_tma


n = [int(m) for m in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]]

# Original tridiagonal matrix algorithm
times_otma = []
for i in n:
    start = time.process_time()
    x, sol, algorithm = run_tma(i)
    end = time.process_time()
    times_otma.append(end-start)
print(times_otma)
