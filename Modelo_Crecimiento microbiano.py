import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def monod_model(y, t, umax, Yxs, S0, Ks):
    X, S = y
    mu = umax * (S / (Ks + S))
    dXdt = X * mu
    dSdt = (-1 / Yxs) * dXdt
    return [dXdt, dSdt]

def gompertz_model(y, t, C, M, B):
    X = y[0]
    dXdt = C * np.exp(-np.exp(-B * (t - M)))
    return dXdt

def monod_growth_rate(X, Yxs, S0, Ks):
    return (Yxs * S0 + X) / (Yxs * S0 + X)

def gompertz_growth_rate(B, C):
    return B * C / np.exp(1)

def monod_latency_duration(M):
    return (M - 1) / monod_growth_rate(M)

def gompertz_latency_duration(M, B):
    return (M - 1) / B

def monod_max_population_density(X0, Yxs, S0, Ks):
    return X0 + (Yxs * S0 + X0) / (Yxs * S0)

def gompertz_max_population_density(A, C):
    return A + C

def integrate_monod_model(umax, Yxs, S0, Ks, X0, t):
    sol = odeint(monod_model, [X0, S0], t, args=(umax, Yxs, S0, Ks))
    X, S = sol.T
    return X, S

def integrate_gompertz_model(C, M, B, X0, t):
    sol = odeint(gompertz_model, [X0], t, args=(C, M, B))
    X = sol.T[0]
    return X

def main():
    print("Select a model:")
    print("1. Monod Model")
    print("2. Gompertz Model")
    model_choice = int(input("Enter 1 or 2: "))

    if model_choice == 1:
        umax = float(input("Enter umax: "))
        Yxs = float(input("Enter Yxs: "))
        S0 = float(input("Enter S0: "))
        Ks = float(input("Enter Ks: "))
        X0 = float(input("Enter initial biomass (X0): "))
        duration = float(input("Enter duration of simulation (hours): "))
        t = np.linspace(0, duration, 1000)

        X, S = integrate_monod_model(umax, Yxs, S0, Ks, X0, t)

        plt.plot(t, X, label="Biomass (X)")
        plt.plot(t, S, label="Substrate (S)")
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration")
        plt.legend()
        plt.show()

        growth_rate = monod_growth_rate(X[-1], Yxs, S0, Ks)
        latency_duration = monod_latency_duration(growth_rate)
        max_population_density = monod_max_population_density(X0, Yxs, S0, Ks)

        print(f"Monod Growth Rate: {growth_rate}")
        print(f"Monod Latency Duration: {latency_duration} hours")
        print(f"Monod Max Population Density: {max_population_density}")

    elif model_choice == 2:
        # Para el modelo de Gompertz
        C = float(input("Enter C (microbial counts when time grows indefinitely, unidades UFC/mL): "))
        M = float(input("Enter M (time to reach maximum specific growth rate, unidades horas): "))
        B = float(input("Enter B (relative growth rate, unidades UFC/mL * horas): "))
        X0 = float(input("Enter initial biomass (X0): "))
        duration = float(input("Enter duration of simulation (hours): "))
        t = np.linspace(0, duration, 1000)

        X = integrate_gompertz_model(C, M, B, X0, t)

        plt.plot(t, X, label="Biomass (X)")
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration")
        plt.legend()
        plt.show()

        growth_rate = gompertz_growth_rate(B, C)
        latency_duration = gompertz_latency_duration(M, B)
        max_population_density = gompertz_max_population_density(X0, C)

        print(f"Gompertz Growth Rate: {growth_rate}")
        print(f"Gompertz Latency Duration: {latency_duration} hours")
        print(f"Gompertz Max Population Density: {max_population_density}")

    else:
        print("Invalid choice. Please enter 1 or 2.")

    continue_choice = input("Do you want to continue? (y/n): ")
    if continue_choice.lower() == 'y':
        main()
    else:
        print("Thank you for using the software.")

if __name__ == "__main__":
    main()
