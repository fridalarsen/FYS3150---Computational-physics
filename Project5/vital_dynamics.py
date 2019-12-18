import numpy as np
import matplotlib.pyplot as plt
from improved_SIR import SIR_model

# Part c - Population A

a = 4.0
b= 1.0
c = 0.5

d = 0.2
d1 = 0.55
e = 0.3

model = SIR_model(400, vital_dynamics=True)

model.set_disease_parameters(a, b, c)
model.vital_rates(e, d, d1)
S, I, R, t = model.solve_RK4(300, 100, 1000, 0.0, 12.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.plot(t, S+I+R, label="Total population", color="rebeccapurple")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/RK4_popA_vr.png')
plt.show()


n = 5000
model = SIR_model(400, vital_dynamics=True)

model.set_disease_parameters(a, b, c)
model.vital_rates(e, d, d1)

S, I, R, t = model.solve_MC(300, 100, int(n), 0.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.plot(t, S+I+R, label="Total population", color="rebeccapurple")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/MC_popA_vr.png')
plt.show()
