import math

def T_verlies_leiding(lengte):
    return 6

T_verlies_leiding_12 = T_verlies_leiding(5)
T_A = 21.8
A = 5  # m²
U = 2000  # W/m²·K
debiet_imec = 60 #m3/h
debiet_backbone = 50 #m3/h
dichtheid_fluid_imec = 997 #kg/m3
dichtheid_fluid_backbone = 997 #kg/m3
m_dot_imec = debiet_imec*dichtheid_fluid_imec/3600  # kg/s
m_dot_backbone = debiet_backbone*dichtheid_fluid_backbone/3600  # kg/s
Cp_fluid_imec = 4180  # J/kg·K
Cp_fluid_backbone = 4180 # J/kg·K

T_1 = 15
T_1_old = 14
while abs(T_1-T_1_old)>0.1:
    T_1=(T_1+T_1_old)/2
    T_2 = T_1 - T_verlies_leiding_12
    T_16=T_2

    C_imec = Cp_fluid_imec*m_dot_imec
    C_backbone = Cp_fluid_backbone*m_dot_backbone
    C_min = min(C_imec,C_backbone)
    C_max = max(C_imec,C_backbone)

    Q_max = C_min*(T_A-T_16)
    C_star = C_min/C_max

    NTU = U*A/C_min
    #gelijkstroom
    #epsilon = (1-math.exp(-NTU*(1+C_star)))/(1+C_star)

    #tegenstroom
    epsilon = (1-math.exp(-NTU*(1-C_star)))/(1-C_star*math.exp(-NTU*(1-C_star)))

    Q = epsilon*Q_max
    T_1_old = T_1
    T_B = T_A - Q/(m_dot_imec*Cp_fluid_imec)
    T_1 = T_16 + Q/(m_dot_backbone*Cp_fluid_backbone)

    print("T_1 =", T_1)
    print("T_16 =", T_16)

print("T_1 =", T_1)





