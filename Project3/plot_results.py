import numpy as np
import matplotlib.pyplot as plt

def read_results(task):
    filename = f"results_{task}.dat"
    N = []
    results = []
    var = []
    errors = []
    times = []
    if task == "3a" or task == "3b":
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines()[1:]):
                components = line.strip().split(" - ")
                times.append(float(components[3]))
                if i % 5 == 0:
                    N.append(int(components[0]))
                    results.append(float(components[1]))
                    errors.append(float(components[2]))
        return N, results, errors, times
    else:
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines()[1:]):
                components = line.strip().split(" - ")
                results.append(float(components[1]))
                var.append(float(components[2]))
                errors.append(float(components[3]))
                times.append(float(components[4]))
                if i % 5 == 0:
                    N.append(int(components[0]))
        N = np.array(N)
        results = np.array(results)
        var = np.array(var)
        errors = np.array(errors)
        times = np.array(times)
        return N, results, var, errors, times

N_a, results_a, deviation_a, times_a = read_results("3a")
times_a = np.array(times_a)
times_a_avg = np.mean(times_a.reshape(-1,5), axis=1)
time_std_a = np.std(times_a.reshape(-1,5), axis=1)

N_b, results_b, deviation_b, times_b = read_results("3b")
times_b = np.array(times_b)
times_b_avg = np.mean(times_b.reshape(-1,5), axis=1)
time_std_b = np.std(times_b.reshape(-1,5), axis=1)

N_c, results_c, var_c, deviation_c, times_c = read_results("3c")
mean_results_c = np.mean(results_c.reshape(-1,5), axis=1)
std_c = np.sqrt(np.mean(var_c.reshape(-1,5), axis=1)/N_c)
times_c_avg = np.mean(times_c.reshape(-1,5), axis=1)
time_std_c = np.std(times_c.reshape(-1,5), axis=1)

N_d, results_d, var_d, deviation_d, times_d = read_results("3d")
mean_results_d = np.mean(results_d.reshape(-1,5), axis=1)
std_d = np.sqrt(np.mean(var_d.reshape(-1,5), axis=1)/N_d)
times_d_avg = np.mean(times_d.reshape(-1,5), axis=1)
time_std_d = np.std(times_d.reshape(-1,5), axis=1)

N_e, results_e, var_e, deviation_e, times_e = read_results("3e")
mean_results_e = np.mean(results_e.reshape(-1,5), axis=1)
std_e = np.sqrt(np.mean(var_e.reshape(-1,5), axis=1)/N_d)
times_e_avg = np.mean(times_e.reshape(-1,5), axis=1)
time_std_e = np.std(times_e.reshape(-1,5), axis=1)

exact = 5*(np.pi**2)/(16**2)

plt.axhline(exact, ls="-", c="darkorange", lw=2, label="Analytic")
plt.plot(N_a, results_a, marker='o', ls="none", ms=6, c="darkred",
             label="Gauss-Legendre")
plt.plot(N_b, results_b, marker='o', ls="none", ms=6, c="darkgreen",
             label="Mixed")
plt.legend()
plt.title("Integral Approximations - Gaussian Quadrature", fontsize=15)
plt.xlabel("N", fontsize=12)
plt.ylabel("I", fontsize=12)
plt.savefig("./Figures/QC_quadrature.png")
plt.show()

plt.axhline(exact, ls="-", c="darkorange", lw=2, label="Analytic")
plt.errorbar(N_c, (mean_results_c), std_c, marker="o", ls="none", c="darkred",
             label="Crude Monte Carlo")
plt.errorbar(N_d, (mean_results_d), std_d, marker="o", ls="none", c="darkgreen",
             label="Improved Monte Carlo")
plt.xscale("log")
plt.legend(loc="upper left")
plt.title("Integral Approximations - Monte Carlo", fontsize=15)
plt.xlabel("N", fontsize=12)
plt.ylabel("I", fontsize=12)
plt.savefig("./Figures/MC.png")
plt.show()

plt.plot(N_a, times_a_avg, marker="o", ls="none", label="Gauss-Legendre",
         c="darkred")
plt.plot(N_b, times_b_avg, marker="o", ls="none", label="Mixed",
         c="darkgreen")
plt.legend()
plt.title("Average Algorithm Times", fontsize=15)
plt.xlabel("N", fontsize=12)
plt.ylabel("t [sek]", fontsize=12)
plt.yscale("log")
plt.savefig("./Figures/algo_times_GQ.png")
plt.show()

plt.errorbar(N_c, times_c_avg, time_std_c, marker="o", ls="none", ms=3,
             label="Crude Monte Carlo", c="darkred")
plt.errorbar(N_d, times_d_avg, time_std_d, marker="o", ls="none", ms=3,
             label="Improved Monte Carlo", c="darkgreen")
plt.errorbar(N_e, times_e_avg, time_std_e, marker="o", ls="none", c="indigo",
             ms=3, label="Parallelized improved Monte Carlo")
plt.title("Average Algorithm Times", fontsize=15)
plt.xlabel("N", fontsize=12)
plt.ylabel("t [sek]", fontsize=12)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("./Figures/algo_time_MC.png")
plt.show()
