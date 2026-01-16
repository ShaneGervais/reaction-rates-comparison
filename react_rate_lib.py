from email.mime import text
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.integrate as integrate
import requests
import pandas as pd
from io import StringIO
import re
from dataclasses import dataclass
from typing import Optional

kB = const.Boltzmann
NA = const.Avogadro
hbar = const.hbar
e = const.e

# ----------------------------
# Helpers
# ----------------------------

def reduced_mass(A1, A2):
    return A1 * A2 / (A1 + A2)

def cross_section(E, Z1, Z2, A1, A2, S):

    mu = reduced_mass(A1, A2)
    nu = np.sqrt(2*E/mu)
    eta = Z1*Z2*e**2/(hbar*nu)
    sigma = S * np.exp(-2*np.pi*eta)/E

    return sigma

def extract_S_factor(df_xs, Z1, Z2, A1, A2):
    """
    Compute astrophysical S-factor from EXFOR cross sections.

    Inputs:
      df_xs["E"]   : energy in eV
      df_xs["SIG"] : cross section in barns

    Returns:
      DataFrame with E_keV and S(E) in keV*barn
    """
    mu = reduced_mass(A1, A2)  # amu

    E_keV = df_xs["E"].to_numpy() * 1e-3
    sigma_b = df_xs["Data"].to_numpy()

    # Remove bad points
    mask = (E_keV > 0) & np.isfinite(sigma_b)
    E_keV = E_keV[mask]
    sigma_b = sigma_b[mask]

    # Sommerfeld parameter (standard astro approximation)
    twopi_eta = 31.29 * Z1 * Z2 * np.sqrt(mu / E_keV)

    S_keVb = sigma_b * E_keV * np.exp(twopi_eta)

    return pd.DataFrame({
        "E_keV": E_keV,
        "SIG_b": sigma_b,
        "S_keVb": S_keVb
    })


def nonres_direct_capture_reaction_rate(E, T, Z1, Z2, A1, A2, S):

    # E -> E-grid over the Gamow window
    mu = reduced_mass(A1, A2)
    I = integrate.simpson(cross_section(E, Z1, Z2, A1, A2, S) * E * np.exp(-E/(kB*T)), E)

    return (8/(np.pi*mu))**(1/2) * (1/(kB*T))**(3/2) * I * NA

def resonant_direct_capture_reaction_rate(T, gamma_omega, A1, A2, Er):

    C = 1.54e11 # constant
    C2 = 11.605 # constant

    return C * (1/(A1 + A2))**(3/2) * gamma_omega * T**(-3/2) * np.exp(-C2 * Er / T)

def total_reaction_rate(T, Z1, Z2, A1, A2, S, gamma_omega, Er):

    R_nonres = nonres_direct_capture_reaction_rate(np.linspace(0.001, 10, 1000)*1.60218e-13, T*1.60218e-10, Z1, Z2, A1, A2, S)
    R_res = resonant_direct_capture_reaction_rate(T, gamma_omega, A1, A2, Er)

    return R_nonres + R_res

def gamow_energy(T, Z1, Z2, A1, A2):

    mu = reduced_mass(A1, A2)
    C1 = 0.122 # constant

    return C1 * (Z1**2 * Z2**2 * mu)**(1/3)* T**(2/3)

def gamow_width(T, Z1, Z2, A1, A2):

    mu = reduced_mass(A1, A2)
    C2 = 0.2368 # constant

    return C2 * (Z1**2 * Z2**2 * mu)**(1/6)*T**(5/6)

def plot_reaction_rate(T_array, rate_array):
    plt.figure(figsize=(8,6))
    plt.plot(T_array, rate_array, label='Total Reaction Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Temperature (GK)')
    plt.ylabel('Reaction Rate (cm³ mol⁻¹ s⁻¹)')
    plt.title('Nuclear Reaction Rate vs Temperature')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


# ----------------------------
# AI GENERATED PARSING CODE
# ----------------------------

_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?$")


def is_num(s: str) -> bool:
    return bool(_NUM.match(s))


@dataclass(frozen=True)
class C4Header:
    tokens: list[str]
    iE: int
    idE: int
    iD: int
    idD: int
    iCos: Optional[int]


def parse_c4_header(text: str) -> C4Header:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") and "Energy" in line and "dEnergy" in line and "Data" in line:
            tokens = line.lstrip("#").split()
            iE = tokens.index("Energy")
            idE = tokens.index("dEnergy")
            iD = tokens.index("Data")
            idD = tokens.index("dData")
            iCos = tokens.index("Cos") if "Cos" in tokens else None
            return C4Header(tokens=tokens, iE=iE, idE=idE, iD=iD, idD=idD, iCos=iCos)
    raise RuntimeError("Could not find C4 header line containing Energy/dEnergy/Data/dData.")


def parse_c4(text: str) -> pd.DataFrame:
    """
    Parse EXFOR C4 output into a DataFrame.
    Keeps the leading metadata columns (Proj/Targ/M/MF/MT/PXC) + numeric fields.

    Returns columns:
      Proj, Targ, M, MF, MT, PXC, E, dE, Data, dData, Cos, raw
    """
    h = parse_c4_header(text)
    header_line_found = False
    rows = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Find the header; start reading after it
        if line.startswith("#") and "Energy" in line and "dEnergy" in line and "Data" in line:
            header_line_found = True
            continue

        if not header_line_found:
            continue

        if line.startswith("#"):
            continue

        parts = line.split()

        # Need at least enough columns to reach numeric indices
        need = max(h.idD, h.iCos or 0)
        if len(parts) <= need:
            continue

        # Numeric guards
        if not (is_num(parts[h.iE]) and is_num(parts[h.idE]) and is_num(parts[h.iD]) and is_num(parts[h.idD])):
            continue

        # Keep front columns if present (common in C4 tables)
        # Typical: Proj Targ M MF MT PXC Energy dEnergy Data dData Cos
        front = parts[:min(6, len(parts))]
        while len(front) < 6:
            front.append(None)

        cos_val = None
        if h.iCos is not None and h.iCos < len(parts) and is_num(parts[h.iCos]):
            cos_val = float(parts[h.iCos])

        rows.append({
            "Proj": front[0],
            "Targ": front[1],
            "M": front[2],
            "MF": front[3],
            "MT": front[4],
            "PXC": front[5],
            "E": float(parts[h.iE]),
            "dE": float(parts[h.idE]),
            "Data": float(parts[h.iD]),
            "dData": float(parts[h.idD]),
            "Cos": cos_val,
            "raw": line,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Parsed 0 numeric data rows from C4. Inspect the raw response header/format.")
    return df


def fetch_exfor_c4(target: str, reaction: str, quantity: str, timeout: int = 30) -> str:
    """
    Fetch C4 output via x4dat (bulk).
    """
    base = "https://nds.iaea.org/exfor"
    params = {
        "target": target,
        "reaction": reaction,
        "quantity": quantity,
        "op": "c4",
    }
    r = requests.get(f"{base}/x4dat", params=params, timeout=timeout,
                     headers={"User-Agent": "reaction-rates-comparison/0.1"})
    r.raise_for_status()
    return r.text


def summarize_blocks(df: pd.DataFrame, n: int = 10):
    """
    Print how many rows appear in each (MF, MT, PXC) block.
    Useful to filter out non-cross-section blocks.
    """
    counts = df[["MF", "MT", "PXC"]].value_counts()
    print("\nTop (MF, MT, PXC) blocks:")
    print(counts.head(n))

def pick_largest_block(df):
    # Pick the most common (MF, MT, PXC) combination as default
    key_counts = df[["MF","MT","PXC"]].value_counts()
    (MF, MT, PXC) = key_counts.index[0]
    return df[(df["MF"] == MF) & (df["MT"] == MT) & (df["PXC"] == PXC)].copy()


def main():

    c4_text = fetch_exfor_c4(target="FE-56", reaction="p,g", quantity="SIG")
    df = parse_c4(c4_text)
    
    print(df.head())
    print(df[["E", "dE", "Data", "dData"]].describe())
    summarize_blocks(df)

    df_xs = pick_largest_block(df)
    df_xs = df_xs.rename(columns={"SIG": "SIG", "dSIG": "dSIG"})
    df_xs = df_xs.sort_values("E")

    # Example parameters
    Z1 = 1  # Proton number of particle 1
    Z2 = 26  # Proton number of particle 2
    A1 = 1  # Atomic mass of particle 1
    A2 = 56  # Atomic mass of particle 2
    S = 1e6 # S-factor in MeV*barn
    gamma_omega = 0.1 # Resonance strength in MeV
    Er = 0.5 # Resonant energy in MeV
    
    Sdf = extract_S_factor(df_xs, Z1, Z2, A1, A2)
    plt.loglog(Sdf["E_keV"], Sdf["S_keVb"], "o")
    plt.xlabel("E (keV)")
    plt.ylabel("S(E) [keV·b]")
    plt.show()

    T_array = np.logspace(-2, 1, 100) # Temperature array from 0.01 to 10 GK
    rate_array = []

    for T in T_array:
        rate = total_reaction_rate(T, Z1, Z2, A1, A2, np.mean(Sdf["S_keVb"]), gamma_omega, Er)
        rate_array.append(rate)

    plot_reaction_rate(T_array, rate_array)

if __name__ == "__main__":
    main()
