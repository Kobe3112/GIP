import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygfunction as gt
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import time
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.colors as mcolors
from scipy.linalg import lstsq
import sys

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;  # Pas deze waarde aan naar de gewenste breedte
        margin: 0 auto;  # Centreer de inhoud
    }
    </style>
    """,
    unsafe_allow_html=True
)

warmtepompen = defaultdict(dict)
HP_data = defaultdict(dict)

#################################################
###### HIER KOMEN ALLE FUNCTIES #################
#################################################
def T_daling_warmtepomp(WP, T_in):

    massadebiet = warmtepompen[WP]["m"]
    T_in_source = T_in
    warmtepompen[WP]["T_in_source"] = T_in_source
    T_req_building = warmtepompen[WP]["T_req_building"]
    Q_req_building = warmtepompen[WP]["Q_req_building"]

    if massadebiet == 0:
        return 0

    if model_WP == "fixed":
        warmtepompen[WP]["T_building"] = T_req_building
        percentage = warmtepompen[WP]["percentage"]
        COP = COP_fixed
        warmtepompen[WP]["COP"] = COP
        Q_building = percentage * Q_req_building
        warmtepompen[WP]["Q_building"] = Q_building
        P_compressor = Q_building / COP
        warmtepompen[WP]["P_compressor"] = P_compressor
        Q_evap = Q_building - P_compressor
        T_out_source = -Q_evap / (massadebiet * Cp_fluid_backbone) + T_in_source
        warmtepompen[WP]["T_out_source"] = T_out_source
        warmtepompen[WP]["delta_T"] = T_in_source - T_out_source

        return T_in_source - T_out_source

    else:
        selected_model = warmtepompen[WP]["selected_model"]
        COP, P_compressor = bereken_variabele_COP_en_P(T_in_source, T_req_building, selected_model, WP)
        warmtepompen[WP]["COP"] = COP
        if selected_model == "Viessmann_biquadratic":
            warmtepompen[WP]["COP_Viessmann_biquadratic"] = COP
        if selected_model == "Viessmann_bilinear":
            warmtepompen[WP]["COP_Viessmann_bilinear"] = COP
        warmtepompen[WP]["P_compressor"] = P_compressor
        Q_building = COP * P_compressor
        warmtepompen[WP]["Q_building"] = Q_building
        warmtepompen[WP]["percentage"] = Q_building/Q_req_building
        delta_T = warmtepompen[WP]["delta_T"]
        massadebiet = Q_building / (delta_T * Cp_fluid_backbone)
        warmtepompen[WP]["m"] = massadebiet
        warmtepompen[WP]["T_in_source"] = T_in
        warmtepompen[WP]["T_out_source"] = T_in - delta_T

        return delta_T

def bereken_variabele_COP_en_P(T_in, T_out, model, WP):

    for i in range(len(model)):
        if model[i-1] == "_":
            HP_model = model[0:i-1]
            if model[i+2] == "l":
                fit = "bilinear"
            if model[i+2] == "q":
                fit = "biquadratic"

    data = HP_data[HP_model]["data"]
    T_max = HP_data[HP_model]["T_max"]

    if T_out > T_max:
        T_out = T_max

    warmtepompen[WP]["T_building"] = T_out

    T_in_data = data[:, 0]
    T_out_data = data[:, 1]
    COP_data = data[:, 2]
    P_data = data[:, 3]

    if fit == "bilinear":

        X = np.column_stack([
            np.ones_like(T_in_data),
            T_in_data,
            T_out_data,
            T_in_data * T_out_data
        ])

        params_COP, residuals, rank, s = lstsq(X, COP_data)
        params_P, residuals, rank, s = lstsq(X, P_data)
        A_COP, B_COP, C_COP, D_COP = params_COP
        A_P, B_P, C_P, D_P = params_P

        COP = A_COP + B_COP * T_in + C_COP * T_out + D_COP * T_in * T_out
        P = A_P + B_P * T_in + C_P * T_out + D_P * T_in * T_out

        return COP, P

    elif fit == "biquadratic":

        X = np.column_stack([
            np.ones_like(T_in_data),
            T_in_data,
            T_out_data,
            T_in_data * T_out_data,
            T_in_data**2,
            T_out_data**2,
            T_in_data**2 * T_out_data,
            T_in_data * T_out_data**2,
            T_in_data**2 * T_out_data**2
        ])

        params_COP, residuals, rank, s = lstsq(X, COP_data)
        params_P, residuals, rank, s = lstsq(X, P_data)
        A_COP, B_COP, C_COP, D_COP, E_COP, F_COP, G_COP, H_COP, I_COP = params_COP
        A_P, B_P, C_P, D_P, E_P, F_P, G_P, H_P, I_P = params_P

        COP = A_COP + B_COP*T_in + C_COP*T_out + D_COP*T_in*T_out + E_COP*T_in**2 + F_COP*T_out**2 + \
            G_COP*T_in**2*T_out + H_COP*T_in*T_out**2 + I_COP*T_in**2*T_out**2
        P = A_P + B_P*T_in + C_P*T_out + D_P*T_in*T_out + E_COP*T_in**2 + F_COP*T_out**2 + \
            G_COP*T_in**2*T_out + H_COP*T_in*T_out**2 + I_COP*T_in**2*T_out**2

        return COP, P
    else:
        sys.exit()

def T_daling_leiding(begin_Temperatuur,lengte,massadebiet):

    if massadebiet == 0:
        return 0

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

    # warmtepomp 10

    T_I10 = T_1
    T_I10_O10 = T_daling_warmtepomp("WP10", T_I10)
    T_O10 = T_I10 - T_I10_O10

    # warmtepomp 1

    T_1_2 = T_daling_leiding(T_1,L_1_2,m_1_2)
    T_2 = T_1 - T_1_2

    T_2_3 = T_daling_leiding(T_2,L_2_3,m_2_3)
    T_3 = T_2 - T_2_3

    T_3_I1 = T_daling_leiding(T_3,L_3_I1,m_WP1)
    T_I1 = T_3 - T_3_I1

    T_I1_O1 = T_daling_warmtepomp("WP1", T_I1)
    T_O1 = T_I1 - T_I1_O1

    T_O1_14 = T_daling_leiding(T_O1,L_O1_14,m_WP1)
    T_14_A = T_O1 - T_O1_14

    # warmtepomp 2

    T_3_I2 = T_daling_leiding(T_3,L_3_I2,m_WP2)
    T_I2 = T_3 - T_3_I2

    T_I2_O2 = T_daling_warmtepomp("WP2", T_I2)
    T_O2 = T_I2 - T_I2_O2

    T_O2_14 = T_daling_leiding(T_O2,L_O2_14,m_WP2)
    T_14_B = T_O2 - T_O2_14

    # retour

    T_14 = meng(T_14_A,m_WP1,T_14_B,m_WP2)

    T_14_T_15 = T_daling_leiding(T_14,L_14_15,m_14_15)
    T_15_B = T_14 - T_14_T_15

    # warmtepomp 3

    T_2_4 = T_daling_leiding(T_2,L_2_4,m_2_4)
    T_4 = T_2 - T_2_4

    T_4_I3 = T_daling_leiding(T_4,L_4_I3,m_WP3)
    T_I3 = T_4 - T_4_I3

    T_I3_O3 = T_daling_warmtepomp("WP3", T_I3)
    T_O3 = T_I3 - T_I3_O3

    T_O3_13 = T_daling_leiding(T_O3,L_O3_13,m_WP3)
    T_13_A = T_O3 - T_O3_13

    # warmtepomp 4

    T_4_5 = T_daling_leiding(T_4, L_4_5, m_4_5)
    T_5 = T_4 - T_4_5

    T_5_I4 = T_daling_leiding(T_5, L_5_I4, m_WP4)
    T_I4 = T_5 - T_5_I4

    T_I4_O4 = T_daling_warmtepomp("WP4", T_I4)
    T_O4 = T_I4 - T_I4_O4

    T_O4_12 = T_daling_leiding(T_O4, L_O4_12, m_WP4)
    T_12_A = T_O4 - T_O4_12

    # warmtepomp 5

    T_5_6 = T_daling_leiding(T_5, L_5_6, m_5_6)
    T_6 = T_5 - T_5_6

    T_6_I5 = T_daling_leiding(T_6, L_6_I5, m_WP5)
    T_I5 = T_6 - T_6_I5

    T_I5_O5 = T_daling_warmtepomp("WP5", T_I5)
    T_O5 = T_I5 - T_I5_O5

    T_O5_11 = T_daling_leiding(T_O5, L_O5_11, m_WP5)
    T_11_A = T_O5 - T_O5_11

    # warmtepomp 6

    T_6_7 = T_daling_leiding(T_6, L_6_7, m_WP6)
    T_7 = T_6 - T_6_7

    T_7_I6 = T_daling_leiding(T_7, L_7_I6, m_WP6)
    T_I6 = T_7 - T_7_I6

    T_I6_O6 = T_daling_warmtepomp("WP6", T_I6)
    T_O6 = T_I6 - T_I6_O6

    T_O6_10 = T_daling_leiding(T_O6, L_O6_10, m_WP6)
    T_10_A = T_O6 - T_O6_10

    # warmtepomp 7

    T_7_8 = T_daling_leiding(T_7,L_7_8,m_7_8)
    T_8 = T_7 - T_7_8

    T_8_I7 = T_daling_leiding(T_8, L_8_I7, m_WP7)
    T_I7 = T_8 - T_8_I7

    T_I7_O7 = T_daling_warmtepomp("WP7", T_I7)
    T_O7 = T_I7 - T_I7_O7

    # warmtepomp 8

    T_8_I8 = T_daling_leiding(T_8, L_8_I8, m_WP8)
    T_I8 = T_8 - T_8_I8

    T_I8_O8 = T_daling_warmtepomp("WP8", T_I8)
    T_O8 = T_I8 - T_I8_O8

    # warmtepomp 9

    T_8_I9 = T_daling_leiding(T_8, L_8_I9, m_WP9)
    T_I9 = T_8 - T_8_I9

    T_I9_O9 = T_daling_warmtepomp("WP9", T_I9)
    T_O9 = T_I9 - T_I9_O9

    # retour

    T_O7_9 = T_daling_leiding(T_O7,L_O7_9,m_WP7)
    T_9_A = T_O7_9

    T_O8_9 = T_daling_leiding(T_O8, L_O8_9, m_WP8)
    T_9_B = T_O8_9

    T_O9_9 = T_daling_leiding(T_O9, L_O9_9, m_WP9)
    T_9_C = T_O9_9

    T_9_1 = meng(T_9_A,m_WP7,T_9_B,m_WP8)
    T_9 = meng(T_9_1,(m_WP7 + m_WP8),T_9_C,m_WP9)

    T_9_10 = T_daling_leiding(T_9,L_9_10,m_9_10)
    T_10_B = T_9 - T_9_10

    T_10 = meng(T_10_A,m_WP6,T_10_B,m_9_10)

    T_10_11 = T_daling_leiding(T_10,L_10_11,m_10_11)
    T_11_B = T_10 - T_10_11

    T_11 = meng(T_11_A,m_WP5,T_11_B,m_WP6)

    T_11_12 = T_daling_leiding(T_11, L_11_12, m_11_12)
    T_12_B = T_11 - T_11_12

    T_12 = meng(T_12_A,m_WP4,T_12_B,m_11_12)

    T_12_13 = T_daling_leiding(T_12,L_12_13,m_12_13)
    T_13_B = T_12 - T_12_13

    T_13 = meng(T_13_A,m_WP3,T_13_B,m_12_13)

    T_13_15 = T_daling_leiding(T_13,L_13_15,m_13_15)
    T_15_A = T_13 - T_13_15

    T_15 = meng(T_15_A,m_2_4,T_15_B,m_14_15)

    T_15_16 = T_daling_leiding(T_15,L_15_16,m_15_16)
    T_16_A = T_15 - T_15_16

    T_16_B = T_O10

    T_16 = meng(T_16_A,m_15_16,T_16_B,m_WP10)

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
    solution['T WP7 IN'] = T_I7
    solution['T WP7 OUT'] = T_O7
    solution['T WP8 IN'] = T_I8
    solution['T WP8 OUT'] = T_O8
    solution['T WP9 IN'] = T_I9
    solution['T WP9 OUT'] = T_O9
    solution['T WP10 IN'] = T_I10
    solution['T WP10 OUT'] = T_O10
    return T_1 - T_16

def bereken_massadebieten_in_leidingen():
    global m_1_2
    global m_2_3
    global m_WP1
    global m_WP2
    global m_2_4
    global m_WP3
    global m_4_5
    global m_WP4
    global m_5_6
    global m_WP5
    global m_6_7
    global m_WP6
    global m_7_8
    global m_WP7
    global m_WP8
    global m_WP9
    global m_9_10
    global m_10_11
    global m_11_12
    global m_12_13
    global m_13_15
    global m_14_15
    global m_15_16
    global m_WP10

    m_1_2 = m_dot_backbone
    m_WP10 = X_WP10 * m_dot_backbone
    m_WP1 = X_WP1 * m_dot_backbone
    m_WP2 = X_WP2 * m_dot_backbone
    m_2_3 = m_WP1 + m_WP2
    m_14_15 = m_2_3

    m_2_4 = m_dot_backbone - m_2_3
    m_WP3 = X_WP3 * m_dot_backbone
    m_4_5 = m_2_4 - m_WP3
    m_WP4 = X_WP4 * m_dot_backbone
    m_5_6 = m_4_5 - m_WP4
    m_WP5 = X_WP5 * m_dot_backbone
    m_6_7 = m_5_6 - m_WP5
    m_WP6 = X_WP6 * m_dot_backbone
    m_WP7 = X_WP7 * m_dot_backbone
    m_WP8 = X_WP8 * m_dot_backbone
    m_WP9 = X_WP9 * m_dot_backbone
    m_7_8 = m_WP7 + m_WP8 + m_WP9
    m_9_10 = m_7_8
    m_10_11 = m_6_7
    m_11_12 = m_5_6
    m_12_13 = m_4_5
    m_13_15 = m_2_4

    m_15_16 = m_14_15 + m_13_15

    if m_15_16 != m_dot_backbone:
        print("Er is een fout me de massadebieten")
def meng(T1,m1,T2,m2):

    if m1 == 0:
        return T2

    if m2 == 0:
        return T1

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
        ((30, 2.9), (29, 2.9),"purple"),
        ((30, 2.9), (30, 1), "purple"),
        ((30, 3.1), (29, 3.1),"orange"),
        ((30, 3.1), (30, 6), "orange"),
        # vertrek uit WW
        ((28, 3.1), (25, 3.1),"red"),
        ((28, 2.9), (25, 2.9),"blue"),
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
        # naar WP 3
        ((25.1, 5.9), (27, 5.9), "blue"),
        ((24.9, 6.1), (27, 6.1), "red"),
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
        ((18.9, 7.1), (18.9, 5.5), "blue"),
        ((19.1, 6.9), (19.1, 5.5), "red"),
        # naar WP 6
        ((16.9, 7.1), (16.9, 6), "blue"),
        ((17.1, 6.9), (17.1, 6), "red"),
        # naar WP 10
        ((26.5, 2.9), (26.5, 3.7), "blue"),
        ((26.3, 3.1), (26.3, 3.7), "red"),
        # return
        ((27.2, 3.1), (27.2, 2.9), "gradient")

    ]
    if WP7_8_9_checkbox:
        leidingen.extend([
            # uiterst links
            ((16.9, 7.1), (14.9, 7.1), "blue"),
            ((17.1, 6.9), (15.1, 6.9), "red"),
            # naar beneden
            ((14.9, 7.1), (14.9, 4), "blue"),
            ((15.1, 6.9), (15.1, 4), "red"),
            # naar WP 9
            ((14.9, 4), (14.9, 2.5), "blue"),
            ((15.1, 4), (15.1, 2.5), "red"),
            # naar WP 8
            ((14.9, 4.1), (16.5, 4.1), "blue"),
            ((15.1, 3.9), (16.5, 3.9), "red"),
            # naar WP 7
            ((14.9, 4.1), (13.5, 4.1), "blue"),
            ((15.1, 3.9), (13.5, 3.9), "red"),
        ])

    for (start, eind, color) in leidingen:
        if color == "gradient":  # Speciale handling voor kleurverloop
            x = np.full(100, start[0])  # x blijft constant (verticale lijn)
            y = np.linspace(start[1], eind[1], 100)  # Opdelen in kleine segmenten

            # Maak een kleurverloop van rood naar blauw
            colors = [mcolors.to_rgba(c) for c in ["red", "blue"]]
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

            for i in range(len(y) - 1):
                ax.plot([x[i], x[i]], [y[i], y[i + 1]], color=cmap(i / len(y)), linewidth=leiding_dikte)

        else:
            ax.plot([start[0], eind[0]], [start[1], eind[1]], color=color, linewidth=leiding_dikte)


    # ⚙️ **Warmtewisselaar**
    ww = patches.Circle((28.5, 3), radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(ww)
    ax.text(28.5, 3, "WW", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")

    imec = patches.Rectangle((29.25, 6), height=0.7, width=1.5, color="orange", ec="black")
    ax.add_patch(imec)
    ax.text(30, 6.35, "IMEC", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")

    dijle = patches.Rectangle((29.25, 1), height=0.7, width=1.5, color="purple", ec="black")
    ax.add_patch(dijle)
    ax.text(30, 1.35, "DIJLE", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")

    # ⚙️ **Warmtepomp**
    pos_WP1 = (23,4.5)
    pos_WP2 = (21,1.5)
    pos_WP3 = (27.5,6)
    pos_WP4 = (21,5.5)
    pos_WP5 = (19,5)
    pos_WP6 = (17,5.5)
    if WP7_8_9_checkbox:
        pos_WP7 = (13,4)
        pos_WP8 = (17,4)
        pos_WP9 = (15,2)
    pos_WP10 = (26.4,4.2)



    WP1 = patches.Circle(pos_WP1, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP1)
    ax.text(pos_WP1[0],pos_WP1[1], "WP1", ha="center", va="center", fontsize=wp_grootte*schaal_factor, fontweight="bold")

    WP2 = patches.Circle(pos_WP2, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP2)
    ax.text(pos_WP2[0], pos_WP2[1], "WP2", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
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
    WP10 = patches.Circle(pos_WP10, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP10)
    ax.text(pos_WP10[0], pos_WP10[1], "WP10", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    if WP7_8_9_checkbox:
        WP7 = patches.Circle(pos_WP7, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP7)
        ax.text(pos_WP7[0], pos_WP7[1], "WP7", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
        WP8 = patches.Circle(pos_WP8, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP8)
        ax.text(pos_WP8[0], pos_WP8[1], "WP8", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
        WP9 = patches.Circle(pos_WP9, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP9)
        ax.text(pos_WP9[0], pos_WP9[1], "WP9", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
    # 🔲 **Temperatuurkaders**
    temperaturen = {
        #WW
        (29.4, 3.3, "orange"): str(round(T_imec,2)) + "°C",
        (29.4, 2.7, "purple"): str(solution["T naar Dijle"]) + "°C",
        (27.5, 3.3, "red"): str(solution["T WARMTEWISSELAAR OUT"]) + "°C",
        (27.5, 2.7, "blue"): str(solution["T WARMTEWISSELAAR IN"]) + "°C",
        # WP1
        (23.65, 3.9, "red"): str(solution["T WP1 IN"]) + "°C",
        (22.35, 3.9, "blue"): str(solution["T WP1 OUT"]) + "°C",
        # WP2
        (20.35, 2.1, "red"): str(solution["T WP2 IN"]) + "°C",
        (21.65, 2.1, "blue"): str(solution["T WP2 OUT"]) + "°C",
        # WP3
        (26.6, 6.4, "red"): str(solution["T WP3 IN"]) + "°C",
        (26.6, 5.6, "blue"): str(solution["T WP3 OUT"]) + "°C",
        # WP4
        (21.65, 6.1, "red"): str(solution["T WP4 IN"]) + "°C",
        (20.35, 6.1, "blue"): str(solution["T WP4 OUT"]) + "°C",
        # WP5
        (19.65, 5.6, "red"): str(solution["T WP5 IN"]) + "°C",
        (18.35, 5.6, "blue"): str(solution["T WP5 OUT"]) + "°C",
        # WP6
        (17.65, 6.1, "red"): str(solution["T WP6 IN"]) + "°C",
        (16.35, 6.1, "blue"): str(solution["T WP6 OUT"]) + "°C",
        # WP10
        (25.8, 3.6, "red"): str(solution["T WP10 IN"]) + "°C",
        (27, 3.6, "blue"): str(solution["T WP10 OUT"]) + "°C",
    }
    if WP7_8_9_checkbox:
        temperaturen.update({
        # WP7
        (14.0, 3.7, "red"): str(solution["T WP7 IN"]) + "°C",
        (14.0, 4.3, "blue"): str(solution["T WP7 OUT"]) + "°C",
        # WP8
        (16, 3.7, "red"): str(solution["T WP8 IN"]) + "°C",
        (16, 4.3, "blue"): str(solution["T WP8 OUT"]) + "°C",
        # WP9
        (15.7, 2.6, "red"): str(solution["T WP9 IN"]) + "°C",
        (14.35, 2.6, "blue"): str(solution["T WP9 OUT"]) + "°C",
        })
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
        ax.text(x, y, temp, ha="center", va="center", fontsize=kader_grootte*schaal_factor+3, fontweight="bold",color=letter_color)

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

modellen = ["Viessmann_bilinear", "Viessmann_biquadratic","Daikin_bilinear", "Daikin_biquadratic"]

### VISUALISATIE INPUT
progress_bar = st.progress(0)
for i in range(101):
    time.sleep(0.01)
    progress_bar.progress(i)
with st.expander("Gegevens invoeren"):
    col1, col2 = st.columns(2)

    with col1:
        T_imec = st.number_input("Temperatuur IMEC", value=21.8, step=0.1)
    with col2:
        debiet_backbone = st.slider("Volumedebiet backbone [m^3/h]", 20, 100, value=70)


    col3, col4 = st.columns(2)
    with col3:
        model_WP = st.selectbox("COP-model", ["vaste waarde", "variabel"])
        #WP7_8_9_checkbox = st.checkbox("Warmtepomp 7-8-9", value=True)
        WP7_8_9_checkbox = st.checkbox("Warmtepomp 7-8-9",value=True)
    with col4:
        if model_WP == "vaste waarde":
            model_WP = 'fixed'
            COP_fixed = st.number_input("Vaste waarde COP:", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    if model_WP == "variabel":
        tab1, tab2 = st.tabs(["Warmtepompen 1-5", "Warmtepompen 6-10"])
        with tab1:
            col5, col6, col7, col8, col9 = st.columns(5)
            with col5:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 1</h4>", unsafe_allow_html=True)
                selected_model_WP1 = st.selectbox("Model", modellen, key ='modelWP1')
                delta_T_WP1 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP1')
            with col6:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 2</h4>", unsafe_allow_html=True)
                selected_model_WP2 = st.selectbox("Model", modellen,key ='modelWP2')
                delta_T_WP2 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP2')
            with col7:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 3</h4>", unsafe_allow_html=True)
                selected_model_WP3 = st.selectbox("Model", modellen,key ='modelWP3')
                delta_T_WP3 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP3')
            with col8:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 4</h4>", unsafe_allow_html=True)
                selected_model_WP4 = st.selectbox("Model", modellen,key ='modelWP4')
                delta_T_WP4 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP4')
            with col9:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 5</h4>", unsafe_allow_html=True)
                selected_model_WP5 = st.selectbox("Model", modellen,key ='modelWP5')
                delta_T_WP5 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP5')
        with tab2:
            col10, col11, col12, col13, col14 = st.columns([1,1,1,1,1])
            with col10:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 6</h4>", unsafe_allow_html=True)
                selected_model_WP6 = st.selectbox("Model", modellen,key ='modelWP6')
                delta_T_WP6 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP6')
            with col11:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 7</h4>", unsafe_allow_html=True)
                selected_model_WP7 = st.selectbox("Model", modellen,key ='modelWP7')
                delta_T_WP7 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP7')
            with col12:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 8</h4>", unsafe_allow_html=True)
                selected_model_WP8 = st.selectbox("Model", modellen,key ='modelWP8')
                delta_T_WP8 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP8')
            with col13:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 9</h4>", unsafe_allow_html=True)
                selected_model_WP9 = st.selectbox("Model", modellen,key ='modelWP9')
                delta_T_WP9 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP9')
            with col14:
                st.markdown("<h4 style='text-align: center; font-size:22px;'>WARMTEPOMP 10</h4>", unsafe_allow_html=True)
                selected_model_WP10 = st.selectbox("Model", modellen,key ='modelWP10')
                delta_T_WP10 = st.number_input("\u0394T bron", min_value=2.0, max_value=8.0, value=5.0, step=0.1,key ='deltaTWP10')

    else:
        selected_model_WP1 = 'Not identified'
        selected_model_WP2 = 'Not identified'
        selected_model_WP3 = 'Not identified'
        selected_model_WP4 = 'Not identified'
        selected_model_WP5 = 'Not identified'
        selected_model_WP6 = 'Not identified'
        selected_model_WP7 = 'Not identified'
        selected_model_WP8 = 'Not identified'
        selected_model_WP9 = 'Not identified'
        selected_model_WP10 = 'Not identified'

        delta_T_WP1 = 0
        delta_T_WP2 = 0
        delta_T_WP3 = 0
        delta_T_WP4 = 0
        delta_T_WP5 = 0
        delta_T_WP6 = 0
        delta_T_WP7 = 0
        delta_T_WP8 = 0
        delta_T_WP9 = 0
        delta_T_WP10 = 0




warmtepompen['WP1']['model'] = selected_model_WP1
warmtepompen['WP2']['model'] = selected_model_WP2
warmtepompen['WP3']['model'] = selected_model_WP3
warmtepompen['WP4']['model'] = selected_model_WP4
warmtepompen['WP5']['model'] = selected_model_WP5
warmtepompen['WP6']['model'] = selected_model_WP6
warmtepompen['WP7']['model'] = selected_model_WP7
warmtepompen['WP8']['model'] = selected_model_WP8
warmtepompen['WP9']['model'] = selected_model_WP9
warmtepompen['WP10']['model'] = selected_model_WP10

warmtepompen['WP1']['selected_model'] = selected_model_WP1
warmtepompen['WP2']['selected_model'] = selected_model_WP2
warmtepompen['WP3']['selected_model'] = selected_model_WP3
warmtepompen['WP4']['selected_model'] = selected_model_WP4
warmtepompen['WP5']['selected_model'] = selected_model_WP5
warmtepompen['WP6']['selected_model'] = selected_model_WP6
warmtepompen['WP7']['selected_model'] = selected_model_WP7
warmtepompen['WP8']['selected_model'] = selected_model_WP8
warmtepompen['WP9']['selected_model'] = selected_model_WP9
warmtepompen['WP10']['selected_model'] = selected_model_WP10

warmtepompen['WP1']['delta_T'] = delta_T_WP1
warmtepompen['WP2']['delta_T'] = delta_T_WP2
warmtepompen['WP3']['delta_T'] = delta_T_WP3
warmtepompen['WP4']['delta_T'] = delta_T_WP4
warmtepompen['WP5']['delta_T'] = delta_T_WP5
warmtepompen['WP6']['delta_T'] = delta_T_WP6
warmtepompen['WP7']['delta_T'] = delta_T_WP7
warmtepompen['WP8']['delta_T'] = delta_T_WP8
warmtepompen['WP9']['delta_T'] = delta_T_WP9
warmtepompen['WP10']['delta_T'] = delta_T_WP10

### WARMTEWISSELAAR
type_WW = 'tegenstroom'
A = 50  # m²
U = 3000  # W/m²·K

### DATA WARMTEPOMPEN

HP_data["Viessmann"]["data"] = np.array([
            [0, 35, 4.3, 13200],
            [45, 90, 3.2, 41200],
            [10, 35, 5.5, 15400]])
HP_data["Viessmann"]["T_max"] = 45

HP_data["Daikin"]["data"] = np.array([
            [0, 35, 4.3, 13200],
            [45, 90, 3.2, 41200],
            [10, 35, 5.5, 15400]])
HP_data["Daikin"]["T_max"] = 45

### WARMTEPOMPEN

# WP1
T_req_building_WP1 = 50  # °C
Q_req_building_WP1 = 200000  # W
percentage_WP1 = 0.70

# WP2
T_req_building_WP2 = 50  # °C
Q_req_building_WP2 = 200000  # W
percentage_WP2 = 0.70

# WP3
T_req_building_WP3 = 50  # °C
Q_req_building_WP3 = 200000  # W
percentage_WP3 = 0.70

# WP4
T_req_building_WP4 = 50  # °C
Q_req_building_WP4 = 200000  # W
percentage_WP4 = 0.70

# WP5
T_req_building_WP5 = 50  # °C
Q_req_building_WP5 = 200000  # W
percentage_WP5 = 0.70

# WP6
T_req_building_WP6 = 50  # °C
Q_req_building_WP6 = 200000  # W
percentage_WP6 = 0.70

# WP7
T_req_building_WP7 = 50  # °C
Q_req_building_WP7 = 200000  # W
percentage_WP7 = 0.70

# WP8
T_req_building_WP8 = 50  # °C
Q_req_building_WP8 = 200000  # W
percentage_WP8 = 0.70

# WP9
T_req_building_WP9 = 50  # °C
Q_req_building_WP9 = 200000  # W
percentage_WP9 = 0.70

# WP10
T_req_building_WP10 = 50  # °C
Q_req_building_WP10 = 200000  # W
percentage_WP10 = 0.70

### FLUIDS
debiet_imec = 60 #m3/h

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
L_7_8 = 375  # m
L_8_I7 = 325  # m
L_8_I8 = 50  # m
L_8_I9 = 50  # m
L_O9_9 = L_8_I9  # m
L_O8_9 = L_8_I8  # m
L_O7_9 = L_8_I7  # m
L_9_10 = L_7_8  # m

### WARMTEPOMPEN DEBIETEN

if WP7_8_9_checkbox:
    X_WP1 = 0.1
    X_WP2 = 0.1
    X_WP3 = 0.1
    X_WP4 = 0.1
    X_WP5 = 0.1
    X_WP6 = 0.1
    X_WP7 = 0.1
    X_WP8 = 0.1
    X_WP9 = 0.1
    X_WP10 = 0.1
else:
    X_WP1 = 0.2
    X_WP2 = 0.2
    X_WP3 = 0.2
    X_WP4 = 0.1
    X_WP5 = 0.1
    X_WP6 = 0.1
    X_WP7 = 0.0
    X_WP8 = 0.0
    X_WP9 = 0.0
    X_WP10 = 0.1

if round(X_WP1 + X_WP2 + X_WP3 + X_WP4 + X_WP5 + X_WP6 + X_WP7 + X_WP8 + X_WP9 + X_WP10,1) == 1:
    bereken_massadebieten_in_leidingen()
else:
    print("Massafracties door warmtepompen zijn samen niet gelijk aan 1")
    exit()

### ALGEMENE WERKING SCRIPT
initial_guess_T_WW_in = 7
iteratie_error_marge = 0.01
aantal_cijfers_na_komma = 2

###################################
##### warmtepompen dictionary #####
###################################
#warmtepompen["WP1"]["COP_Viessmann_biquadratic"] = 'not defined'
#warmtepompen["WP1"]["COP_Viessmann_bilineair"] = 'not defined'


parameters = ["selected_model","delta_T","m","T_in_source","T_out_source","T_building","T_req_building","Q_building","Q_req_building", "percentage", "P_compressor"]
for i in range(1, 11):
    WP = "WP"+str(i)
    print(WP)
    for j in range(len(parameters)):
        name_var = parameters[j] + "_" + WP
        value = globals().get(name_var)
        warmtepompen[WP][parameters[j]] = value

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
show_solution()
####################################
###### SOLUTION VISUAL ############# ==> IN PROGRESS, NOG NIKS VAN AANTREKKEN
####################################


leiding_dikte = 2  # Dikte van leidingen
kader_grootte = 0.5  # Grootte temperatuurkaders
wp_grootte = 1  # Warmtepomp-grootte
schaal_factor = 25
st.title("WARMTENET")
st.pyplot(teken_schema(solution))

zoek_waarde = st.toggle("Zoek waarde", value=False)

if zoek_waarde:
    # Kader
    with st.container():
        st.markdown("#### Selecteer opties:")

        col1, col2 = st.columns(2)
        with col1:
            selected_wp = st.selectbox("Kies een warmtepomp:", list(warmtepompen.keys()))

        # Dynamisch ophalen van eigenschappen die alleen bij de geselecteerde warmtepomp horen
        beschikbare_eigenschappen = list(warmtepompen[selected_wp].keys())

        with col2:
            selected_property = st.selectbox("Kies een eigenschap:", beschikbare_eigenschappen)

        # Output in hetzelfde kader
        st.markdown(f"**{selected_property} van {selected_wp}:**")
        st.info(f"{warmtepompen[selected_wp][selected_property]}")  # Mooie opmaak voor output

    st.divider()  # Voegt een scheidingslijn toe

for WP in warmtepompen:
    warmtepompen[WP]['T in'] = warmtepompen[WP]['T_in_source']
for WP in warmtepompen:
    warmtepompen[WP]['T out'] = warmtepompen[WP]['T_out_source']

warmtepompen['WP1']['Q_boiler'] = warmtepompen['WP1']['Q_req_building'] - warmtepompen['WP1']['Q_building']
warmtepompen['WP2']['Q_boiler'] = warmtepompen['WP2']['Q_req_building'] - warmtepompen['WP2']['Q_building']
warmtepompen['WP3']['Q_boiler'] = warmtepompen['WP3']['Q_req_building'] - warmtepompen['WP3']['Q_building']
warmtepompen['WP4']['Q_boiler'] = warmtepompen['WP4']['Q_req_building'] - warmtepompen['WP4']['Q_building']
warmtepompen['WP5']['Q_boiler'] = warmtepompen['WP5']['Q_req_building'] - warmtepompen['WP5']['Q_building']
warmtepompen['WP6']['Q_boiler'] = warmtepompen['WP6']['Q_req_building'] - warmtepompen['WP6']['Q_building']
warmtepompen['WP7']['Q_boiler'] = warmtepompen['WP7']['Q_req_building'] - warmtepompen['WP7']['Q_building']
warmtepompen['WP8']['Q_boiler'] = warmtepompen['WP8']['Q_req_building'] - warmtepompen['WP8']['Q_building']
warmtepompen['WP9']['Q_boiler'] = warmtepompen['WP9']['Q_req_building'] - warmtepompen['WP9']['Q_building']
warmtepompen['WP10']['Q_boiler'] = warmtepompen['WP10']['Q_req_building'] - warmtepompen['WP10']['Q_building']

warmtepompen['WP1']['T_boiler'] = warmtepompen['WP1']['T_req_building'] - warmtepompen['WP1']['T_building']
warmtepompen['WP2']['T_boiler'] = warmtepompen['WP2']['T_req_building'] - warmtepompen['WP2']['T_building']
warmtepompen['WP3']['T_boiler'] = warmtepompen['WP3']['T_req_building'] - warmtepompen['WP3']['T_building']
warmtepompen['WP4']['T_boiler'] = warmtepompen['WP4']['T_req_building'] - warmtepompen['WP4']['T_building']
warmtepompen['WP5']['T_boiler'] = warmtepompen['WP5']['T_req_building'] - warmtepompen['WP5']['T_building']
warmtepompen['WP6']['T_boiler'] = warmtepompen['WP6']['T_req_building'] - warmtepompen['WP6']['T_building']
warmtepompen['WP7']['T_boiler'] = warmtepompen['WP7']['T_req_building'] - warmtepompen['WP7']['T_building']
warmtepompen['WP8']['T_boiler'] = warmtepompen['WP8']['T_req_building'] - warmtepompen['WP8']['T_building']
warmtepompen['WP9']['T_boiler'] = warmtepompen['WP9']['T_req_building'] - warmtepompen['WP9']['T_building']
warmtepompen['WP10']['T_boiler'] = warmtepompen['WP10']['T_req_building'] - warmtepompen['WP10']['T_building']
data = {
    'WP': ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6'],
    'T WP in (bron) [°C]': [
        str(warmtepompen['WP1']['T in']), str(warmtepompen['WP2']['T in']), str(warmtepompen['WP3']['T in']),
        str(warmtepompen['WP4']['T in']), str(warmtepompen['WP5']['T in']), str(warmtepompen['WP6']['T in'])
    ],
    'T WP out (bron) [°C]': [
        str(warmtepompen['WP1']['T out']), str(warmtepompen['WP2']['T out']), str(warmtepompen['WP3']['T out']),
        str(warmtepompen['WP4']['T out']), str(warmtepompen['WP5']['T out']), str(warmtepompen['WP6']['T out'])
    ],
    'debiet [m^3/h]' : [
        str(warmtepompen['WP1']['m']), str(warmtepompen['WP2']['m']), str(warmtepompen['WP3']['m']),
        str(warmtepompen['WP4']['m']), str(warmtepompen['WP5']['m']), str(warmtepompen['WP6']['m'])
    ],
    'COP' : [
        str(warmtepompen['WP1']['COP']), str(warmtepompen['WP2']['COP']), str(warmtepompen['WP3']['COP']),
        str(warmtepompen['WP4']['COP']), str(warmtepompen['WP5']['COP']), str(warmtepompen['WP6']['COP'])
    ],
    '\u0394T bron [°C]' : [
        str(warmtepompen['WP1']['delta_T']), str(warmtepompen['WP2']['delta_T']), str(warmtepompen['WP3']['delta_T']),
        str(warmtepompen['WP4']['delta_T']), str(warmtepompen['WP5']['delta_T']), str(warmtepompen['WP6']['delta_T'])
    ]
}

if model_WP == 'variabel':
    data['Model'] = [
        selected_model_WP1, selected_model_WP2, selected_model_WP3,
        selected_model_WP4, selected_model_WP5, selected_model_WP6
    ]

df = pd.DataFrame(data)

if WP7_8_9_checkbox:
    wp789_data = {
        'WP': ['WP7', 'WP8', 'WP9'],
        'T WP in (bron) [°C]': [
            str(warmtepompen['WP7']['T in']), str(warmtepompen['WP8']['T in']), str(warmtepompen['WP9']['T in'])
        ],
        'T WP out (bron) [°C]': [
            str(warmtepompen['WP7']['T out']), str(warmtepompen['WP8']['T out']), str(warmtepompen['WP9']['T out'])
        ],
        'debiet [m^3/h]': [
            str(warmtepompen['WP7']['m']), str(warmtepompen['WP8']['m']), str(warmtepompen['WP9']['m'])
        ],
        '\u0394T bron [°C]' : [
        str(warmtepompen['WP7']['delta_T']), str(warmtepompen['WP8']['delta_T']), str(warmtepompen['WP9']['delta_T'])
        ],
        'COP': [
            str(warmtepompen['WP7']['COP']), str(warmtepompen['WP8']['COP']), str(warmtepompen['WP9']['COP'])
        ]
    }
    if model_WP == 'variabel':
        wp789_data['Model'] = [
            selected_model_WP7, selected_model_WP8, selected_model_WP9
        ]

    wp789_df = pd.DataFrame(wp789_data)
    df = pd.concat([df, wp789_df], ignore_index=True)

wp10_data = {
    'WP': ['WP10'],
    'T WP in (bron) [°C]': [
        str(warmtepompen['WP10']['T in'])
    ],
    'T WP out (bron) [°C]': [
        str(warmtepompen['WP10']['T out'])
    ],
    'debiet [m^3/h]' : [
        str(warmtepompen['WP10']['m'])
    ],
    '\u0394T bron [°C]' : [
        str(warmtepompen['WP10']['delta_T'])
    ],
    'COP' : [
        str(warmtepompen['WP10']['COP'])
    ]

}
if model_WP == 'variabel':
    wp10_data['Model'] = [selected_model_WP10]

wp10_df = pd.DataFrame(wp10_data)
df = pd.concat([df, wp10_df], ignore_index=True)

desired_order = ['T WP in (bron) [°C]', 'T WP out (bron) [°C]']
if '\u0394T bron [°C]' in df.columns:
    desired_order.append('\u0394T bron [°C]')
desired_order.append('COP')
desired_order.append('debiet [m^3/h]')
if 'Model' in df.columns:
    desired_order.append('Model')
df.set_index('WP', inplace=True)
df = df[desired_order]
st.dataframe(df,use_container_width=True)

###############
#df_transposed = df.transpose()
#st.dataframe(df_transposed,use_container_width=True)
#st.markdown(
#    df.style.set_properties(**{'text-align': 'center'}).to_html(),
#    unsafe_allow_html=True
#)
###############

def teken_grafiek():
    # Initialiseer session_state voor tijd en data als ze nog niet bestaan
    if 'tijd' not in st.session_state:
        st.session_state['tijd'] = []
    if 'T_WP1_IN_values' not in st.session_state:
        st.session_state['T_WP1_IN_values'] = []

    # Creëer een placeholder voor de grafiek zonder sleutel
    graph_placeholder = st.empty()

    # Zet een starttijd voor de grafiek
    start_time = time.time()

    # Maak de figuur
    fig = go.Figure()

    # Voeg de eerste lijn toe (om de grafiek te initiëren)
    fig.add_trace(go.Scatter(
        x=st.session_state['tijd'],
        y=st.session_state['T_WP1_IN_values'],
        mode='lines+markers',
        name='T WP1 IN',
        line=dict(color='blue')
    ))

    # Titels en labels toevoegen
    fig.update_layout(
        title="Live data van T WP1 IN",
        xaxis_title="Tijd",
        yaxis_title="Temperatuur [°C]",
        template='plotly_dark',
        xaxis=dict(tickformat="%H:%M:%S")  # Dit format geeft tijd weer in uur:minuut:seconde
    )

    # Toon de grafiek voor het eerst zonder `plotly_chart` aan te roepen
    graph_placeholder.plotly_chart(fig, use_container_width=True, key="unique_graph_key_initial")

    # Loop voor het continu updaten van de grafiek
    while True:
        # Verkrijg de huidige tijd
        current_time = datetime.now()

        # Voeg de tijdstap toe aan de lijst (max 1 keer per seconde)
        if len(st.session_state['tijd']) == 0 or (time.time() - start_time >= 1):
            st.session_state['tijd'].append(current_time)
            start_time = time.time()  # Reset starttijd voor de volgende stap

            # Controleer of de waarde van T WP1 IN is veranderd
            if len(st.session_state['T_WP1_IN_values']) > 0 and solution['T WP1 IN'] != \
                    st.session_state['T_WP1_IN_values'][-1]:
                # Als de waarde is veranderd, voeg de nieuwe waarde toe
                st.session_state['T_WP1_IN_values'].append(solution['T WP1 IN'])
            else:
                # Anders voeg de vorige waarde opnieuw toe (voor horizontale lijn)
                st.session_state['T_WP1_IN_values'].append(
                    st.session_state['T_WP1_IN_values'][-1] if len(st.session_state['T_WP1_IN_values']) > 0 else
                    solution['T WP1 IN'])

        # Voeg de nieuwe data toe aan de grafiek zonder de grafiek opnieuw te tekenen
        fig.data[0].x = st.session_state['tijd']
        fig.data[0].y = st.session_state['T_WP1_IN_values']

        # Genereer een dynamische sleutel op basis van de huidige tijd in milliseconden (zorgt voor uniekheid)
        dynamic_key = f"unique_graph_key_{int(datetime.now().timestamp() * 1000)}"  # Milliseconden

        # Werk de grafiek bij met de nieuwe sleutel
        graph_placeholder.plotly_chart(fig, use_container_width=True, key=dynamic_key)

        # Slaap 1 seconde voor de volgende update
        time.sleep(0.5)

#teken_grafiek()



warmtepompen_list = ["WP1", "WP2", "WP3", "WP4", "WP5", "WP6"]
wp_warmte = [warmtepompen['WP1']['Q_building'],
             warmtepompen['WP2']['Q_building'],
             warmtepompen['WP3']['Q_building'],
             warmtepompen['WP4']['Q_building'],
             warmtepompen['WP5']['Q_building'],
             warmtepompen['WP6']['Q_building']
             ]
boiler_warmte = [warmtepompen['WP1']['Q_boiler'],
                 warmtepompen['WP2']['Q_boiler'],
                 warmtepompen['WP3']['Q_boiler'],
                 warmtepompen['WP4']['Q_boiler'],
                 warmtepompen['WP5']['Q_boiler'],
                 warmtepompen['WP6']['Q_boiler']
             ]
wp_T = [warmtepompen['WP1']['T_building'],
             warmtepompen['WP2']['T_building'],
             warmtepompen['WP3']['T_building'],
             warmtepompen['WP4']['T_building'],
             warmtepompen['WP5']['T_building'],
             warmtepompen['WP6']['T_building']
             ]
boiler_T = [warmtepompen['WP1']['T_boiler'],
                 warmtepompen['WP2']['T_boiler'],
                 warmtepompen['WP3']['T_boiler'],
                 warmtepompen['WP4']['T_boiler'],
                 warmtepompen['WP5']['T_boiler'],
                 warmtepompen['WP6']['T_boiler']
             ]
if WP7_8_9_checkbox:
    warmtepompen_list.extend(["WP7", "WP8", "WP9"])
    wp_warmte.extend([warmtepompen['WP7']['Q_building'],
                      warmtepompen['WP8']['Q_building'],
                      warmtepompen['WP9']['Q_building']])
    boiler_warmte.extend([warmtepompen['WP7']['Q_boiler'],
                          warmtepompen['WP8']['Q_boiler'],
                          warmtepompen['WP9']['Q_boiler']])
    wp_T.extend([warmtepompen['WP7']['T_building'],
                warmtepompen['WP8']['T_building'],
                warmtepompen['WP9']['T_building']])
    boiler_T.extend([warmtepompen['WP7']['T_boiler'],
                    warmtepompen['WP8']['T_boiler'],
                    warmtepompen['WP9']['T_boiler']])

warmtepompen_list.extend(["WP10"])
wp_warmte.extend([warmtepompen['WP10']['Q_building']])
boiler_warmte.extend([warmtepompen['WP10']['Q_boiler']])
wp_warmte.extend([warmtepompen['WP10']['T_building']])
boiler_warmte.extend([warmtepompen['WP10']['T_boiler']])

fig = go.Figure()

fig.add_trace(go.Bar(
    x=warmtepompen_list,
    y=wp_warmte,
    name="Warmtepomp",
    marker_color="green"
))

fig.add_trace(go.Bar(
    x=warmtepompen_list,
    y=boiler_warmte,
    name="Boiler",
    marker_color="orange"
))

# Layout instellingen
fig.update_layout(
    barmode="stack",  # Stapelen van staven
    title="Warmtelevering per gebouw",
    xaxis_title="Warmtepomp",
    yaxis_title="Warmte [Wh]",
    legend_title="Bron van warmte"
)

# Toon de grafiek in Streamlit
st.plotly_chart(fig)


fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=warmtepompen_list,
    y=wp_T,
    name="Warmtepomp",
    marker_color="green"
))

fig2.add_trace(go.Bar(
    x=warmtepompen_list,
    y=boiler_T,
    name="Boiler",
    marker_color="orange"
))

# Layout instellingen
fig2.update_layout(
    barmode="stack",  # Stapelen van staven
    title="Temperatuur aanvoer gebouwkant",
    xaxis_title="Warmtepomp",
    yaxis_title="T [°C]",
    legend_title="Bron van T"
)

# Toon de grafiek in Streamlit
st.plotly_chart(fig2)
#         streamlit run Main.py
