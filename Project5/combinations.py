import numpy as np
import matplotlib.pyplot as plt
from improved_SIR import SIR_model

# Population A

b = 1.0
c = 0.5

d = 0.3
d1 = 0.55
e = 0.3

def f(t):
    if t < 5.0:
        return 0
    else:
        return 0.2

model = SIR_model(400, vital_dynamics=True, vaccination=True, seasonal_variation=True)
model.set_disease_parameters(b=b, c=c)
model.variational_param(3.0, 0.5, 4.0)
model.vital_rates(e, d, d1)
model.vaccination_rate(f)

S, I, R, t = model.solve_RK4(300, 100, 1000, 0.0, 24.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.plot(t, S+I+R, label="Total population", color="rebeccapurple")
plt.axvline(5.0, linestyle="--", color="gray")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/RK4_popA_combo.png')
plt.show()


n = 10000
model = SIR_model(400, vital_dynamics=True, vaccination=True, seasonal_variation=True)

model.set_disease_parameters(b=b, c=c)
model.variational_param(3.0, 0.5, 4.0)
model.vital_rates(e, d, d1)
model.vaccination_rate(f)

S, I, R, t = model.solve_MC(300, 100, int(n), 0.0)

plt.plot(t, S, label="Susceptible", color="crimson")
plt.plot(t, I, label="Infected", color="forestgreen")
plt.plot(t, R, label="Recovered", color="gold")
plt.plot(t, S+I+R, label="Total population", color="rebeccapurple")
plt.axvline(5.0, linestyle="--", color="gray")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of individuals", fontsize=12)
plt.title(f'Disease evolution in population A', fontsize=15)
plt.legend()
plt.savefig('Figures/MC_popA_combo.png')
plt.show()
