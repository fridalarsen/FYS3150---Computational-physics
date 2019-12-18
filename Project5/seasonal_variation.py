import numpy as np
import matplotlib.pyplot as plt
from improved_SIR import SIR_model

# Part d - Population A

b = 1.0
c = 0.5

model = SIR_model(400, seasonal_variation=True)
model.set_disease_parameters(b=b, c=c)
model.variational_param(3.0, 0.5, 4.0)

S, I, R, t = model.solve_RK4(300, 100, 1000, 0.0, 12.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/RK4_popA_sv.png')
plt.show()

S, I, R, t = model.solve_MC(300, 100, 5840, 0.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/MC_popA_sv.png')
plt.show()
