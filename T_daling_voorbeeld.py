import pygfunction as gt
import math

# Instellingen voor de stroming en fluid
debiet = 70             # m³/h
rho = 997               # kg/m³ (water)
vel = 2                 # m/s
m_flow = debiet * rho / 3600  # massastroom (kg/s)

# Bereken de binnenradius (r_in) uit de continuïteitsvergelijking
A = m_flow / (rho * vel)          # doorsnede (m²)
r_in = math.sqrt(A / math.pi)     # binnenradius (m)
print("Binnenradius r_in: {:.4f} m".format(r_in))

# Fluid eigenschappen bij 20°C
mu = 0.001          # dynamische viscositeit (Pa·s)
k_fluid = 0.598     # warmtegeleidingscoëfficiënt (W/m·K)
cp = 4180           # soortelijke warmte (J/kg·K)
epsilon = 0.00005   # ruwheidswaarde (m)

# Bereken de convectieve warmtetransfercoëfficiënt in de pijp
h_conv = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
    m_flow_pipe=m_flow,
    r_in=r_in,
    mu_f=mu,
    rho_f=rho,
    k_f=k_fluid,
    cp_f=cp,
    epsilon=epsilon
)
print("h_conv: {:.1f} W/m²·K".format(h_conv))

# Pijpgeometrie
wanddikte = 0.01           # wanddikte (m)
r_out = r_in + wanddikte    # buitenradius (m)
k_pijp = 45                 # thermische geleidbaarheid pijp (W/m·K)

# Bereken de interne convectieve weerstand
R_conv = 1 / (h_conv * 2 * math.pi * r_in)

# Bereken de leidingweerstand van de pijpwand
R_pijp = math.log(r_out / r_in) / (2 * math.pi * k_pijp)

# Vereenvoudigde berekening van de grondweerstand:
# R_grond = 1/(2*pi*k_grond) * ln(r_eff/r_out)
k_grond = 1.5             # warmtegeleidingscoëfficiënt grond (W/m·K)
r_eff = 1               # diepte begraving leiding (m) --> aanpassen?
R_grond = 1 / (2 * math.pi * k_grond) * math.log(r_eff / r_out)

# Totale thermische weerstand per meter
R_tot = R_conv + R_pijp + R_grond

print("R_conv: {:.4f} K/W".format(R_conv))
print("R_pijp: {:.4f} K/W".format(R_pijp))
print("R_grond: {:.4f} K/W".format(R_grond))
print("Totale weerstand R_tot: {:.4f} K/W".format(R_tot))

# Warmteverlies per meter (ΔT = 20°C - 10°C)
delta_T = 20 - 10       # °C
Q_per_meter = delta_T / R_tot
print("Warmteverlies per meter: {:.2f} W/m".format(Q_per_meter))

# Temperatuurdaling in de vloeistof per meter
delta_T_fluid = Q_per_meter / (m_flow * cp)
print("Temperatuurdaling vloeistof: {:.5f} K/m".format(delta_T_fluid))
