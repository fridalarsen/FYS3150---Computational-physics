import numpy as np
import matplotlib.pyplot as plt
from improved_SIR import SIR_model

# Part e - Population A

a = 4.0
b = 1.0
c = 0.5
def f(t):
    if t < 5.0:
        return 0
    else:
        return 1.0

model = SIR_model(400, vaccination=True)
model.set_disease_parameters(a, b, c)
model.vaccination_rate(f)

S, I, R, t = model.solve_RK4(300, 100, 1000, 0.0, 12.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.axvline(5.0, linestyle="--", color="gray")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/RK4_popA_vacc.png')
plt.show()

S, I, R, t = model.solve_MC(300, 100, 5000, 0.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.axvline(5.0, linestyle="--", color="gray")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/MC_popA_vacc.png')
plt.show()
