import math
#import streamlit as st
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import random
import pygfunction as gt


#################################################
###### HIER KOMEN ALLE FUNCTIES #################
#################################################
def T_daling_warmtepomp(T_cold_in):
    # T_cold_in is de T van het water die de WP binnenkomt aan de koude kant (=backbone side)
    # T_cold_out is de T van het water die de WP uitstroomt aan de koude kant (=backbone side)
    if T_cold_in == T_I1:
        percentage = percentage_WP_1
        max_heat_demand = max_heat_demand_WP_1
        T_hot_out = T_hot_out_WP_1
        massadebiet = m_WP_1
        tekst = "WP1"

    elif T_cold_in == T_I2:
        percentage = percentage_WP_2
        max_heat_demand = max_heat_demand_WP_2
        T_hot_out = T_hot_out_WP_2
        massadebiet = m_WP_2
        tekst = "WP2"

    elif T_cold_in == T_I3:
        percentage = percentage_WP_3
        max_heat_demand = max_heat_demand_WP_3
        T_hot_out = T_hot_out_WP_3
        massadebiet = m_WP_3
        tekst = "WP3"

    elif T_cold_in == T_I4:
        percentage = percentage_WP_4
        max_heat_demand = max_heat_demand_WP_4
        T_hot_out = T_hot_out_WP_4
        massadebiet = m_WP_4
        tekst = "WP4"

    elif T_cold_in == T_I5:
        percentage = percentage_WP_5
        max_heat_demand = max_heat_demand_WP_5
        T_hot_out = T_hot_out_WP_5
        massadebiet = m_WP_5
        tekst = "WP5"

    elif T_cold_in == T_I6:
        percentage = percentage_WP_6
        max_heat_demand = max_heat_demand_WP_6
        T_hot_out = T_hot_out_WP_6
        massadebiet = m_WP_6
        tekst = "WP6"

    else:
        print("Warmtepomp niet gevonden")
        exit()

    COP = bereken_COP(T_cold_in,T_hot_out,model_WP)
    Q_cond = percentage * max_heat_demand
    P_compressor = Q_cond / COP
    Q_evap = Q_cond - P_compressor
    T_cold_out = -Q_evap / (massadebiet * Cp_fluid_backbone) + T_cold_in
    P_compressor_WP[tekst] = P_compressor
    return T_cold_in - T_cold_out
def bereken_COP(T_cold_in, T_hot_out, model):
    if model == "fixed":
        return (4.5)
def T_daling_leiding(begin_Temperatuur,lengte,massadebiet):
    # neem constante snelheid aan doorheen de buizen
    pipe_diameter = math.sqrt((4*massadebiet)/(math.pi*dichtheid_fluid_backbone*flowspeed_backbone))

    Pr = (mu_fluid_backbone*Cp_fluid_backbone)/k_fluid_backbone
    Re = (flowspeed_backbone*pipe_diameter*dichtheid_fluid_backbone)/mu_fluid_backbone

    h_conv = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe=massadebiet,
        r_in=pipe_diameter/2,
        mu_f=mu_fluid_backbone,
        rho_f=dichtheid_fluid_backbone,
        k_f=k_fluid_backbone,
        cp_f=Cp_fluid_backbone,
        epsilon=epsilon_steel)

    R_conv = (math.pi*pipe_diameter)/h_conv
    R_steel = math.log((0.5*pipe_diameter+pipe_thickness)/(0.5*pipe_diameter)) / (2*math.pi*k_steel)
    R_ground = math.log((4 * depth) / pipe_diameter) / (2 * math.pi * k_ground)
    R_tot = R_conv + R_steel + R_ground

    T_diff_tot = 0
    T_in = begin_Temperatuur
    for i in range(lengte):
        q = (T_in - T_ground) / R_tot
        T_diff = q / (Cp_fluid_backbone * massadebiet)
        T_diff_tot = T_diff_tot + T_diff
        T_in = T_in - T_diff
        print(T_in)
    return T_diff_tot

def T_daling_totaal(T_1):
    global T_I1
    global T_I2
    global T_I3
    global T_I4
    global T_I5
    global T_I6
    global T_I7
    global T_I8
    global T_I9

    ##############################################################

    T_1_2 = T_daling_leiding(T_1,L_1_2,m_1_2)
    T_2 = T_1 - T_1_2

    # wartepomp 1

    T_2_3 = T_daling_leiding(T_2,L_2_3,m_2_3)
    T_3 = T_2 - T_2_3

    T_3_I1 = T_daling_leiding(T_3,L_3_I1,m_WP_1)
    T_I1 = T_3 - T_3_I1

    T_I1_O1 = T_daling_warmtepomp(T_I1)
    T_O1 = T_I1 - T_I1_O1

    T_O1_14 = T_daling_leiding(T_O1,L_O1_14,m_WP_1)
    T_14_A = T_O1 - T_O1_14

    # warmtepomp 2

    T_3_I2 = T_daling_leiding(T_3,L_3_I2,m_WP_2)
    T_I2 = T_3 - T_3_I2

    T_I2_O2 = T_daling_warmtepomp(T_I2)
    T_O2 = T_I2 - T_I2_O2

    T_O2_14 = T_daling_leiding(T_O2,L_O2_14,m_WP_2)
    T_14_B = T_O2 - T_O2_14

    # retour

    T_14 = meng(T_14_A,m_WP_1,T_14_B,m_WP_2)

    T_14_T_15 = T_daling_leiding(T_14,L_14_15,m_14_15)
    T_15_B = T_14 - T_14_T_15

    # warmtepomp 3

    T_2_4 = T_daling_leiding(T_2,L_2_4,m_2_4)
    T_4 = T_2 - T_2_4

    T_4_I3 = T_daling_leiding(T_4,L_4_I3,m_WP_3)
    T_I3 = T_4 - T_4_I3

    T_I3_O3 = T_daling_warmtepomp(T_I3)
    T_O3 = T_I3 - T_I3_O3

    T_O3_13 = T_daling_leiding(T_O3,L_O3_13,m_WP_3)
    T_13_A = T_O3 - T_O3_13

    # warmtepomp 4

    T_4_5 = T_daling_leiding(T_4, L_4_5, m_4_5)
    T_5 = T_4 - T_4_5

    T_5_I4 = T_daling_leiding(T_5, L_5_I4, m_WP_4)
    T_I4 = T_5 - T_5_I4

    T_I4_O4 = T_daling_warmtepomp(T_I4)
    T_O4 = T_I4 - T_I4_O4

    T_O4_12 = T_daling_leiding(T_O4, L_O4_12, m_WP_4)
    T_12_A = T_O4 - T_O4_12

    # warmtepomp 5

    T_5_6 = T_daling_leiding(T_5, L_5_6, m_5_6)
    T_6 = T_5 - T_5_6

    T_6_I5 = T_daling_leiding(T_6, L_6_I5, m_WP_5)
    T_I5 = T_6 - T_6_I5

    T_I5_O5 = T_daling_warmtepomp(T_I5)
    T_O5 = T_I5 - T_I5_O5

    T_O5_11 = T_daling_leiding(T_O5, L_O5_11, m_WP_5)
    T_11_A = T_O5 - T_O5_11

    # warmtepomp 6

    T_6_7 = T_daling_leiding(T_6, L_6_7, m_WP_6)
    T_7 = T_6 - T_6_7

    T_7_I6 = T_daling_leiding(T_7, L_7_I6, m_WP_6)
    T_I6 = T_7 - T_7_I6

    T_I6_O6 = T_daling_warmtepomp(T_I6)
    T_O6 = T_I6 - T_I6_O6

    T_O6_10 = T_daling_leiding(T_O6, L_O6_10, m_WP_6)
    T_10 = T_O6 - T_O6_10

    # retour

    T_10_11 = T_daling_leiding(T_10,L_10_11,m_WP_6)
    T_11_B = T_10 - T_10_11

    T_11 = meng(T_11_A,m_WP_5,T_11_B,m_WP_6)

    T_11_12 = T_daling_leiding(T_11, L_11_12, m_11_12)
    T_12_B = T_11 - T_11_12

    T_12 = meng(T_12_A,m_WP_4,T_12_B,m_11_12)

    T_12_13 = T_daling_leiding(T_12,L_12_13,m_12_13)
    T_13_B = T_12 - T_12_13

    T_13 = meng(T_13_A,m_WP_3,T_13_B,m_12_13)

    T_13_15 = T_daling_leiding(T_13,L_13_15,m_13_15)
    T_15_A = T_13 - T_13_15

    T_15 = meng(T_15_A,m_2_4,T_15_B,m_14_15)

    T_15_16 = T_daling_leiding(T_15,L_15_16,m_15_16)
    T_16 = T_15 - T_15_16

    ##############################################################

    solution['T2'] = T_2
    solution['T3'] = T_3
    solution['T4'] = T_4
    solution['T14'] = T_14
    solution['T15'] = T_15
    solution['T WP1 IN'] = T_I1
    solution['T WP1 OUT'] = T_O1
    solution['T WP2 IN'] = T_I2
    solution['T WP2 OUT'] = T_O2
    solution['T WP3 IN'] = T_I3
    solution['T WP3 OUT'] = T_O3
    solution['T WP4 IN'] = T_I4
    solution['T WP4 OUT'] = T_O4
    solution['T WP5 IN'] = T_I5
    solution['T WP5 OUT'] = T_O5
    solution['T WP6 IN'] = T_I6
    solution['T WP6 OUT'] = T_O6
    return T_1 - T_16

def bereken_massadebieten_in_leidingen():
    global m_1_2
    global m_2_3
    global m_WP_1
    global m_WP_2
    global m_2_4
    global m_WP_3
    global m_4_5
    global m_WP_4
    global m_5_6
    global m_WP_5
    global m_WP_6
    global m_11_12
    global m_12_13
    global m_13_15
    global m_14_15
    global m_15_16

    m_1_2 = m_dot_backbone
    m_WP_1 = X_WP1 * m_dot_backbone
    m_WP_2 = X_WP2 * m_dot_backbone
    m_2_3 = m_WP_1 + m_WP_2
    m_14_15 = m_2_3

    m_2_4 = m_dot_backbone - m_2_3
    m_WP_3 = X_WP3 * m_dot_backbone
    m_4_5 = m_2_4 - m_WP_3
    m_WP_4 = X_WP4 * m_dot_backbone
    m_5_6 = m_4_5 - m_WP_4
    m_WP_5 = X_WP5 * m_dot_backbone
    m_WP_6 = X_WP6 * m_dot_backbone
    m_11_12 = m_5_6
    m_12_13 = m_4_5
    m_13_15 = m_2_4

    m_15_16 = m_14_15 + m_13_15

    if m_15_16 != m_dot_backbone:
        print("Er is een fout me de massadebieten")
def meng(T1,m1,T2,m2):
    T_new = (T1*m1+T2*m2)/(m1+m2)
    return T_new
def itereer_over_volledig_netwerk():

    T_16 = initial_guess_T_WW_in
    T_16_old = initial_guess_T_WW_in - iteratie_error_marge - 1

    C_imec = Cp_fluid_imec * m_dot_imec
    C_backbone = Cp_fluid_backbone * m_dot_backbone
    C_min = min(C_imec, C_backbone)
    C_max = max(C_imec, C_backbone)

    while abs(T_16 - T_16_old) > iteratie_error_marge:
        Q_max = C_min * (T_imec - T_16)
        C_star = C_min / C_max
        NTU = U * A / C_min

        if type_WW == 'gelijkstroom':
            epsilon = (1 - math.exp(-NTU * (1 + C_star))) / (1 + C_star)
        elif type_WW == 'tegenstroom':
            epsilon = (1 - math.exp(-NTU * (1 - C_star))) / (1 - C_star * math.exp(-NTU * (1 - C_star)))
        else:
            epsilon = 0

        # andere optie
        # epsilon=(1 / C_star) * (1 - np.exp(-C_star * (1 - np.exp(-NTU))))

        Q = epsilon * Q_max
        T_16_old = T_16
        T_naar_Dijle = T_imec - Q / (m_dot_imec * Cp_fluid_imec)
        T_1 = T_16 + Q / (m_dot_backbone * Cp_fluid_backbone)
        T_16 = T_1 - T_daling_totaal(T_1)

    solution['T WARMTEWISSELAAR OUT'] = T_1
    solution['T WARMTEWISSELAAR IN'] = T_16
    solution['T naar Dijle'] = T_naar_Dijle

def teken_schema(solution):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect('equal')  # Zorgt ervoor dat cirkels geen ovalen worden
    x_min, x_max = ax.get_xlim()
    schaal_factor = (x_max - x_min) / 10  # Hoe groter de figuur, hoe groter de tekst

    # 🏭 **Leidingen**
    leidingen = [
        ((1, 3), (4, 3)),  # Leiding naar warmtepomp
        ((5, 3), (8, 3)),  # Leiding na warmtepomp
    ]

    for (start, eind) in leidingen:
        ax.plot([start[0], eind[0]], [start[1], eind[1]], color="black", linewidth=leiding_dikte)

    # ⚙️ **Warmtepomp**
    warmtepomp = patches.Circle((4.5, 3), radius=wp_grootte / 2, color="lightblue", ec="black")
    ax.add_patch(warmtepomp)
    ax.text(4.5, 3, "WP", ha="center", va="center", fontsize=letter_grootte*schaal_factor*1.5, fontweight="bold")

    # 🔲 **Temperatuurkaders**
    temperaturen = {
        (1.2, 3.2): str(solution["T1"])+"°C",  # Begin leiding 1
        (3.8, 3.2): str(solution["T2"])+"°C",  # Einde leiding 1
        (5.2, 3.2): str(solution["T1"])+"°C",  # Warmtepomp uitgang
        (7.8, 3.2): str(solution["T16"])+"°C"  # Einde leiding 2
    }

    for (x, y), temp in temperaturen.items():
        rect = patches.Rectangle((x - kader_grootte / 2, y - 0.2), kader_grootte, kader_grootte, color="white", ec="black")
        ax.add_patch(rect)
        ax.text(x, y, temp, ha="center", va="center", fontsize=letter_grootte*schaal_factor, fontweight="bold")

    # 🔧 **Lay-out instellingen**

    ax.set_adjustable("datalim")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return fig

##################################################
###### INPUT PARAMETERS/AANNAMES #################
##################################################

### WARMTEWISSELAAR
type_WW = 'tegenstroom'
A = 50  # m²
U = 3000  # W/m²·K


### WARMTEPOMPEN
model_WP = "fixed"

max_heat_demand_WP_1 = 250000  # W
percentage_WP_1 = 0.70
T_hot_out_WP_1 = 50 #°C

max_heat_demand_WP_2 = 200000  # W
percentage_WP_2 = 0.70
T_hot_out_WP_2 = 40 #°C

max_heat_demand_WP_3 = 200000  # W
percentage_WP_3 = 0.70
T_hot_out_WP_3 = 40 #°C

max_heat_demand_WP_4 = 200000  # W
percentage_WP_4 = 0.70
T_hot_out_WP_4 = 40 #°C

max_heat_demand_WP_5 = 200000  # W
percentage_WP_5 = 0.70
T_hot_out_WP_5 = 40 #°C

max_heat_demand_WP_6 = 200000  # W
percentage_WP_6 = 0.70
T_hot_out_WP_6 = 40 #°C

### FLUIDS
T_imec = 21.8
debiet_imec = 60 #m3/h
debiet_backbone = 70 #m3/h
dichtheid_fluid_imec = 997 #kg/m3
dichtheid_fluid_backbone = 997 #kg/m3
Cp_fluid_imec = 4180  # J/kg·K
Cp_fluid_backbone = 4180  # J/kg·K
k_fluid_backbone = 0.598  # W/(m*k)
mu_fluid_backbone = 0.001  # N*s/m2

m_dot_imec = debiet_imec * dichtheid_fluid_imec / 3600  # kg/s
m_dot_backbone = debiet_backbone * dichtheid_fluid_backbone / 3600  # kg/s

### LEIDINGEN ONTWERPDATA
depth = 1  # m
k_ground = 1  # W/(m*K)
pipe_thickness = 0.01  # m
k_steel = 45  # W/(m*K)
epsilon_steel = 0.00005   # m
T_ground = 10  # °C
flowspeed_backbone = 2  # m/s

### LEIDINGEN LENGTES
L_1_2 = 115  # m
L_2_3 = 30  # m
L_3_I1 = 20  # m
L_O1_14 = L_3_I1  # m
L_3_I2 = 75  # m
L_O2_14 = L_3_I2  # m
L_14_15 = L_2_3  # m
L_15_16 = L_1_2  # m
L_2_4 = 60  # m
L_4_I3 = 115  # m
L_O3_13 = L_4_I3  # m
L_13_15 = L_2_4  # m
L_4_5 = 190  # m
L_5_I4 = 40  # m
L_O4_12 = L_5_I4  # m
L_5_6 = 50  # m
L_6_I5 = 25  # m
L_O5_11 = L_6_I5  # m
L_6_7 = 100  # m
L_7_I6 = 25  # m
L_O6_10 = L_7_I6  # m
L_10_11 = L_6_7  # m
L_11_12 = L_5_6  # m
L_12_13 = L_4_5  # m

### WARMTEPOMPEN DEBIETEN
X_WP1 = 0.2
X_WP2 = 0.2
X_WP3 = 0.2
X_WP4 = 0.2
X_WP5 = 0.1
X_WP6 = 0.1
X_WP7 = 0.0
X_WP8 = 0.0
X_WP9 = 0.0

if round(X_WP1 + X_WP2 + X_WP3 + X_WP4 + X_WP5 + X_WP6 + X_WP7 + X_WP8 + X_WP9,1) == 1:
    bereken_massadebieten_in_leidingen()
else:
    print("Massafracties door warmtepompen zijn samen niet gelijk aan 1")
    exit()


### ALGEMENE WERKING SCRIPT
initial_guess_T_WW_in = 5
iteratie_error_marge = 0.1
aantal_cijfers_na_komma = 2

####################################
###### START SCRIPT ################
####################################

solution = {}
P_compressor_WP={}
itereer_over_volledig_netwerk()
for Temperatuur in solution:
    solution[Temperatuur] = round(solution[Temperatuur], aantal_cijfers_na_komma)

####################################
###### SHOW SOLUTION ###############
####################################
print("-----------------------------------------------")
print("Dit zijn alle temperaturen:")
solution_sorted = dict(sorted(solution.items()))
for Temperatuur, value in solution_sorted.items():
    print("*",f"{Temperatuur}: {value}","°C")
print("-----------------------------------------------")
print("En dit zijn de compressor vermogens:")
P_compressor_WP_sorted = dict(sorted(P_compressor_WP.items()))
for WP, value in P_compressor_WP_sorted.items():
    print("*",f"{WP}: {round(value/1000)}","kW")

####################################
###### SOLUTION VISUAL ############# ==> IN PROGRESS, NOG NIKS VAN AANTREKKEN
####################################

leiding_dikte = 2  # Dikte van leidingen
kader_grootte = 0.5  # Grootte temperatuurkaders
wp_grootte = 1  # Warmtepomp-grootte
letter_grootte = 60  # Tekstgrootte
#T_A = st.number_input("Temperatuur IMEC",value=21.8,step=0.1)
#st.title("Warmtenet Visualisatie")
#st.pyplot(teken_schema(solution))

