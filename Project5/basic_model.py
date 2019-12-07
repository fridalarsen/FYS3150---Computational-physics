import numpy as np
import matplotlib.pyplot as plt
from improved_SIR import SIR_model

# Part a
a = 4.0
b = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
c = 0.5

model = SIR_model(400)
for key, b_val in b.items():
    model.set_disease_parameters(a, b_val, c)
    S, I, R, t = model.solve_RK4(300, 100, 1000, 0.0, 12.0)

    plt.plot(t, S, label="Susceptible", color="crimson")
    plt.plot(t, I, label="Infected", color="forestgreen")
    plt.plot(t, R, label="Recovered", color="gold")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Number of individuals", fontsize=12)
    plt.title(f'Disease evolution in population {key}', fontsize=15)
    plt.legend()
    plt.savefig(f'Figures/RK4_population_{key}')
    plt.show()

# Part b
for key, b_val in b.items():
    n = 5000*b_val
    model.set_disease_parameters(a, b_val, c)

    S, I, R, t = model.solve_MC(300, 100, int(n), 0.0)

    plt.plot(t, S, label="Susceptible", color="crimson")
    plt.plot(t, I, label="Infected", color="forestgreen")
    plt.plot(t, R, label="Recovered", color="gold")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Number of individuals", fontsize=12)
    plt.title(f'Disease evolution in population {key}', fontsize=15)
    plt.legend()
    plt.show()
