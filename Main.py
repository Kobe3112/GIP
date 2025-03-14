import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


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
        return (COP_fixed)
def T_daling_leiding(begin_Temperatuur):
    if begin_Temperatuur>20:
        return 1
    elif begin_Temperatuur>15:
        return 0.5 + random.uniform(-0.01,0.01)
    elif begin_Temperatuur>10:
        return 0
    else:
        return -0.5
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


    T_1_2 = T_daling_leiding(T_1)
    T_2 = T_1 - T_1_2

    T_2_3 = T_daling_leiding(T_2)
    T_3 = T_2 - T_2_3

    T_3_I1 = T_daling_leiding(T_3)
    T_I1 = T_3 - T_3_I1

    T_I1_O1 = T_daling_warmtepomp(T_I1)
    T_O1 = T_I1 - T_I1_O1

    T_O1_14 = T_daling_leiding(T_O1)
    T_14_A = T_O1 - T_O1_14

    T_3_I2 = T_daling_leiding(T_3)
    T_I2 = T_3 - T_3_I2

    T_I2_O2 = T_daling_warmtepomp(T_I2)
    T_O2 = T_I2 - T_I2_O2

    T_O2_14 = T_daling_leiding(T_O2)
    T_14_B = T_O2 - T_O2_14

    T_14 = meng(T_14_A,m_WP_1,T_14_B,m_WP_2)

    T_14_T_15 = T_daling_leiding(T_14)
    T_15_B = T_14 - T_14_T_15

    #om het nu te doen werken ##############
    T_2_15 = 15
    T_15_A = T_2 - T_2_15
    #####################


    T_15 = meng(T_15_A,m_2_15,T_15_B,m_14_15)

    T_15_16 = T_daling_leiding(T_15)
    T_16 = T_15 - T_15_16

    solution['T2'] = T_2
    solution['T3'] = T_3
    solution['T14'] = T_14
    solution['T15'] = T_15
    solution['T WP1 IN'] = T_I1
    solution['T WP1 OUT'] = T_O1
    solution['T WP2 IN'] = T_I2
    solution['T WP2 OUT'] = T_O2
    return T_1 - T_16
def bereken_massadebieten_in_leidingen():
    global m_1_2
    global m_2_3
    global m_WP_1
    global m_WP_2
    global m_14_15
    global m_13_15
    global m_15_16



    # om het nu te doen werken ############
    global m_2_15
    ###################

    m_1_2 = m_dot_backbone
    m_WP_1 = X_WP1 * m_dot_backbone
    m_WP_2 = X_WP2 * m_dot_backbone
    m_2_3 = m_WP_1 + m_WP_2
    m_14_15 = m_WP_1 + m_WP_2


    # om het nu te doen werken #############"""
    m_2_15 = m_dot_backbone - m_2_3
    ###########################


    m_15_16 = m_14_15 + m_2_15

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
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.set_aspect('equal')  # Zorgt ervoor dat cirkels geen ovalen worden
    x_min, x_max = ax.get_xlim()


    # 🏭 **Leidingen**
    leidingen = [
        # naar WW
        ((30, 2.9), (28, 2.9),"lightblue"),
        ((30, 3.1), (28, 3.1),"orange"),
        # vertrek uit WW
        ((27, 3.1), (25, 3.1),"red"),
        ((27, 2.9), (25, 2.9),"blue"),
        # naar links
        ((25, 3.1), (23, 3.1),"red"),
        ((25, 2.9), (23, 2.9),"blue"),
        # naar WP1
        ((23.1, 3.1), (23.1, 4),"red"),
        ((22.9, 2.9), (22.9, 4),"blue"),
        # naar links
        ((23, 3.1), (20.9, 3.1),"red"),
        ((23, 2.9), (21.1, 2.9),"blue"),
        # naar WP2
        ((21.1, 2.9), (21.1, 2),"blue"),
        ((20.9, 3.1), (20.9, 2),"red"),
        # naar boven
        ((25.1, 2.9), (25.1, 6), "blue"),
        ((24.9, 3.1), (24.9, 6), "red"),

        # nog naar boven
        ((25.1, 6), (25.1, 7.1), "blue"),
        ((24.9, 6), (24.9, 6.9), "red"),
        # bovenleiding naar WP6
        ((25.1, 7.1), (16.9, 7.1), "blue"),
        ((24.9, 6.9), (17.1, 6.9), "red"),
        # naar WP 4
        ((20.9, 7.1), (20.9, 6), "blue"),
        ((21.1, 6.9), (21.1, 6), "red"),
        # naar WP 5
        ((18.9, 7.1), (18.9, 6), "blue"),
        ((19.1, 6.9), (19.1, 6), "red"),
        # naar WP 6
        ((16.9, 7.1), (16.9, 6), "blue"),
        ((17.1, 6.9), (17.1, 6), "red"),

    ]
    if WP3_checkbox:
        leidingen.extend([
            # naar WP 3
            ((25.1, 5.9), (27, 5.9), "blue"),
            ((24.9, 6.1), (27, 6.1), "red"),
        ])

    for (start, eind, color) in leidingen:
        ax.plot([start[0], eind[0]], [start[1], eind[1]], color=color, linewidth=leiding_dikte)

    # ⚙️ **Warmtewisselaar**
    ww = patches.Circle((27.5, 3), radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(ww)
    ax.text(27.5, 3, "WW", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")


    # ⚙️ **Warmtepomp**
    pos_WP1 = (23,4.5)
    pos_WP2 = (21,1.5)
    if WP3_checkbox:
        pos_WP3 = (27.5,6)
    pos_WP4 = (21,5.5)
    pos_WP5 = (19,5.5)
    pos_WP6 = (17,5.5)
    pos_WP7 = ()
    pos_WP8 = ()
    pos_WP9 = ()



    WP1 = patches.Circle(pos_WP1, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP1)
    ax.text(pos_WP1[0],pos_WP1[1], "WP1", ha="center", va="center", fontsize=wp_grootte*schaal_factor, fontweight="bold")

    WP2 = patches.Circle(pos_WP2, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP2)
    ax.text(pos_WP2[0], pos_WP2[1], "WP2", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    if WP3_checkbox:
        WP3 = patches.Circle(pos_WP3, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP3)
        ax.text(pos_WP3[0], pos_WP3[1], "WP3", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
    WP4 = patches.Circle(pos_WP4, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP4)
    ax.text(pos_WP4[0], pos_WP4[1], "WP4", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP5 = patches.Circle(pos_WP5, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP5)
    ax.text(pos_WP5[0], pos_WP5[1], "WP5", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP6 = patches.Circle(pos_WP6, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP6)
    ax.text(pos_WP6[0], pos_WP6[1], "WP6", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")

    # 🔲 **Temperatuurkaders**
    temperaturen = {
        #WW
        (28.1, 3.5, "orange"): str(round(T_imec,2)) + "°C",
        (28.1, 2.5, "lightblue"): str(solution["T naar Dijle"]) + "°C",
        (26.9, 3.5, "red"): str(solution["T WARMTEWISSELAAR OUT"]) + "°C",
        (26.9, 2.5, "blue"): str(solution["T WARMTEWISSELAAR IN"]) + "°C",
        # WP1
        (23.6, 3.9, "red"): str(solution["T WP1 IN"]) + "°C",
        (22.4, 3.9, "blue"): str(solution["T WP1 OUT"]) + "°C",
        # WP2
        (20.4, 2.1, "red"): str(solution["T WP2 IN"]) + "°C",
        (21.6, 2.1, "blue"): str(solution["T WP2 OUT"]) + "°C",
        # WP3
        (26.8, 6.4, "red"): str(solution["T WP3 IN"]) + "°C",
        (26.8, 5.6, "blue"): str(solution["T WP3 OUT"]) + "°C",
        # WP4
        (21.6, 6.1, "red"): str(solution["T WP4 IN"]) + "°C",
        (20.4, 6.1, "blue"): str(solution["T WP4 OUT"]) + "°C",
        # WP5
        (19.6, 6.1, "red"): str(solution["T WP4 IN"]) + "°C",
        (18.4, 6.1, "blue"): str(solution["T WP4 OUT"]) + "°C",
        # WP6
        (17.6, 6.1, "red"): str(solution["T WP4 IN"]) + "°C",
        (16.4, 6.1, "blue"): str(solution["T WP4 OUT"]) + "°C",
    }
    drempel = 10
    for (x, y, letter_color), temp in temperaturen.items():
        waarde = float(temp.replace("°C", ""))  # Extract waarde uit string
        #if waarde < drempel:
            # ❄️ Koud water → Blauw
            #facecolor = "lightblue"
            #edgecolor = "blue"
            #letter_color = "blue"
        #else:
            # 🔥 Warm water → Rood
            #facecolor = "lightcoral"
            #edgecolor = "red"
            #letter_color = "red"
        #rect = patches.Rectangle((x - kader_grootte / 2, y - 0.2), kader_grootte, kader_grootte, color=facecolor, ec=edgecolor)
        #ax.add_patch(rect)
        ax.text(x, y, temp, ha="center", va="center", fontsize=kader_grootte*schaal_factor, fontweight="bold",color=letter_color)

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
model_WP = st.selectbox("COP-model", ["fixed", "model2"])
#model_WP = "fixed"
COP_fixed = 4

max_heat_demand_WP_1 = 250000  # W
percentage_WP_1 = 0.70
T_hot_out_WP_1 = 50 #°C

max_heat_demand_WP_2 = 200000  # W
percentage_WP_2 = 0.70
T_hot_out_WP_2 = 40 #°C

### FLUIDS
#T_imec = 21.8
T_imec = st.number_input("Temperatuur IMEC",value=21.8,step=0.1)
debiet_backbone = st.slider("Volumedebiet backbone [m^3/h]", 20, 80, value=70)
debiet_imec = 60 #m3/h
#debiet_backbone = 50 #m3/h
dichtheid_fluid_imec = 997 #kg/m3
dichtheid_fluid_backbone = 997 #kg/m3
Cp_fluid_imec = 4180  # J/kg·K
Cp_fluid_backbone = 4180  # J/kg·K

m_dot_imec = debiet_imec * dichtheid_fluid_imec / 3600  # kg/s
m_dot_backbone = debiet_backbone * dichtheid_fluid_backbone / 3600  # kg/s

### LEIDINGEN ONTWERPDATA

### LEIDINGEN LENGTES
L_1_2 = 0 # m
L_2_3 = 0 # m

### WARMTEPOMPEN DEBIETEN
X_WP1 = 0.15
X_WP2 = 0.15
X_WP3 = 0.15
X_WP4 = 0.15
X_WP5 = 0.15
X_WP6 = 0.1
X_WP7 = 0.05
X_WP8 = 0.05
X_WP9 = 0.05

if round(X_WP1 + X_WP2 + X_WP3 + X_WP4 + X_WP5 + X_WP6 + X_WP7 + X_WP8 + X_WP9,1) == 1:
    bereken_massadebieten_in_leidingen()
else:
    print("Massafracties door warmtepompen zijn samen niet gelijk aan 1")
    exit()

WP3_checkbox = st.checkbox("Warmtepomp 3")

### ALGEMENE WERKING SCRIPT
initial_guess_T_WW_in = 7
iteratie_error_marge = 0.01
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

def show_solution():
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
#show_solution()
####################################
###### SOLUTION VISUAL ############# ==> IN PROGRESS, NOG NIKS VAN AANTREKKEN
####################################

solution["T WP3 IN"] = 0
solution["T WP3 OUT"] = 0
solution["T WP4 IN"] = 0
solution["T WP4 OUT"] = 0
solution["T WP5 IN"] = 0
solution["T WP5 OUT"] = 0
solution["T WP6 IN"] = 0
solution["T WP6 OUT"] = 0



leiding_dikte = 2  # Dikte van leidingen
kader_grootte = 0.5  # Grootte temperatuurkaders
wp_grootte = 1  # Warmtepomp-grootte
schaal_factor = 30
st.title("Warmtenet Visualisatie")
st.pyplot(teken_schema(solution))

#         streamlit run Main.py

