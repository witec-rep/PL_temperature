import streamlit as st
import matplotlib.pyplot as plt
from AJ_draw import disegna as ds
import base64
import pandas as pd

from AJ_workflow_heatmap import mapping as mp
from AJ_loop_lib import loop
from AJ_PL_analisi import data_storage
from AJ_workflow_preprocess import preprocess
import numpy as np

import gspread
from oauth2client.service_account import  ServiceAccountCredentials

from scipy.optimize import curve_fit
from scipy.stats.distributions import t as tstud

from typing import Dict
import zipfile

# dict_param = {'PL_name':2, 'numb_of_spect':3, 'num_of_scan':4, 'bkg_position':5,
#          'x_bkg':6, 'y_bkg':7, 'remove_cosmic_ray':8, 'soglia_derivata':9,
#          'wave_inf':10, 'wave_sup':11, 'num_bin':12, 'raggio':13,
#          'laser_power':14, 'AS_min':15, 'AS_max':16, 'S_min':17,
#          'S_max':18, 'riga_y_start':19, 'riga_y_stop':20, 'pixel_0_x':21,
#          'pixel_0_y':22, 'laser_type':23, 'lato_cella':24, 'temp_max':25,
#          'cut_min':26, 'cut_max':1, 'punti_cross_section_geometrica':2, 'num_loop1':3,
#          'num_loop2':3, 'soglia_r2':4, 'selected_scan':5, 'T_RT':6}

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def download_file(data, filename):
    testo = 'Download '+filename+'.csv'
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">'+testo+'</a>'
    st.markdown(href, unsafe_allow_html=True)

@st.cache
def load_database():
    sheet11 = client.open("database").get_worksheet(0)
    data1 = sheet11.get_all_records()
    df = pd.DataFrame(data1)

    colonne = df['PL_name'].tolist()
    df = df.T
    df.columns = colonne
    return df

def save_database(index, vet_param):
    if vet_param[3] == 'min':
        vet_param[3] = 0
    else:
        vet_param[3] = 1

    if vet_param[6] == 'yes':
        vet_param[6] = 0
    else:
        vet_param[6] = 1

    sheet11 = client.open("database").get_worksheet(0)
    sheet11.update_cell(index+2, 1, index+1)
    for i in range(len(vet_param)):
        sheet11.update_cell(index+2, i+2, vet_param[i])

def delete_database(data, name):
    indice_del = data[name]['index']
    sheet11 = client.open("database").get_worksheet(0)
    sheet11.delete_row(indice_del + 1)
    new_ind = [i+1 for i in range(len(data.loc['index']) - 1)]
    for id in new_ind:
        sheet11.update_cell(id+1, 1, id)

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)

st.title("Temperature Cruncher")
# password = st.text_input("Enter Password", type = "password")
# password = 'passperte'
# if password == 'passperte':

df = load_database()
exp_names = df.columns
exp_param = df.index


# ██      ███████  ██████  ███████ ███    ██ ██████
# ██      ██      ██       ██      ████   ██ ██   ██
# ██      █████   ██   ███ █████   ██ ██  ██ ██   ██
# ██      ██      ██    ██ ██      ██  ██ ██ ██   ██
# ███████ ███████  ██████  ███████ ██   ████ ██████


if st.checkbox('Type of analysis Legend'):
    st.markdown('**Single**: analyze the map with a fixed **T reference** and a fixed **selected scan**.')
    st.markdown('**Learning on Single**: analyze the map with a fixed **selected scan**, and modify the **T reference** to match the room temperature.')
    st.markdown('**loop numerator**: analyze the map looping on all possible numerators (**seleceted scan**) and all possible **T reference**.')
    st.markdown('**T averaging**: ones the analysis is done, the T average takes the _power_vs_temp_ matrix as input to produce the average plot. (The **loop numerator** does it automatically).')
    st.markdown('**Derivative threshold estimator**: plot the derivative of all the spectrum allowing to chose a threshold.')
    st.markdown('**set all parameters**: display all the parameters for the analysis.')
    st.markdown('**Compare multi particles**: read multiple files with the intesity and the slope and plot a compareson.')
    st.markdown('**Compare log vs standard**: read the T average from 2 files and plot both on the same graph.')
    st.markdown('**Global analysis**: run **loop numeator** with all 3 possible equation and plot all the results.')
if st.checkbox('Parameters Legend'):
    st.markdown('**Name file** (_PL_name_): given name for a set of parameters, it is used to name the output files from the software.')
    st.markdown('**Num of Spectrum** (_numb_of_spect_): Number of spectrum you are going to end up after the binning.')
    st.markdown('**Num of scan along a single axe** (_num_of_scan_): The software expect square maps, here is requested the number of scans on a single edge.')
    st.markdown('**bkg** (_bkg_position_): It gives 2 choices (min, max). With "min" the whole map is subtracted with the sprectrum with the smallest intensity. With "max", the whole map is subtracted with a spectrum from a chosen position. The postion is selected with the parameters **X bkg pixel** and **Y bkg pixel**.')
    st.markdown('**X bkg pixel** (_x_bkg_): X position on the heatmap of the spectrum used for the background.')
    st.markdown('**Y bkg pixel** (_y_bkg_): Y position on the heatmap of the spectrum used for the background.')
    st.markdown('**Remove cosmic ray** (_remove_cosmic_ray_): It gives 2 choices (yes, no). With "yes" an algorithm which remove the cosmic rays noise is run. With "no" it is not.')
    st.markdown('**Derivative threshold** (_derivative_threshold_): threshold used to remove the cosmic ray noise. It can be estimate with the **Derivative threshold estimator**.')
    st.markdown('**Lower limit to calculate the stokes on the heat map** (_wave_inf_): Set the lower limit of the Stokes region of the spectrum to calculate the average intensity. Used on the heatmap, a different parameter is used on the temperature analysis.')
    st.markdown('**Upper limit to calculate the stokes on the heat map** (_wave_sup_): Set the upper limit of the Stokes region of the spectrum to calculate the average intensity. Used on the heatmap, a different parameter is used on the temperature analysis.')
    st.markdown('**Num bin** (_num_bin_): Number of spectrum binned together.')
    st.markdown('**Radius** (_radius_): Radius of the particle analyzed.')
    st.markdown('**Laser power** (_laser_power_): power of the laser on the sample.')
    st.markdown('**AS_min** (_AS_min_): Set the lower limit of the Anti-Stokes region for the analysis.')
    st.markdown('**AS_max** (_AS_max_): Set the upper limit of the Anti-Sotkes region for the analysis.')
    st.markdown('**S_min** (_S_min_): Set the lower limit of the Stoke region for the analysis.')
    st.markdown('**S_max** (_S_max_): Set the upper limit of the Stokes region for the analysis.')
    st.markdown('**Y start delete** (_row_y_start_delete_): some rows can be erased from the analysis, this is the lower Y in the range of the rows to delete.')
    st.markdown('**Y stop delete** (_row_y_stop_delete_): upper Y in the range of rows to delete.')
    st.markdown('**X pixel for delete** (_pixel_for_delete_x_): X coordinate of the spectrum used to replace those in the range **Y start delete - Y stop delete**.')
    st.markdown('**Y pixel for delete** (_pixel_for_delete_y_): Y coordinate of the spectrum used to replace those in the range **Y start delete - Y stop delete**.')
    st.markdown('**Laser wavelength** (_laser_type_): wavelength of the laser used for the experiment.')
    st.markdown('**single pixel size** (_pixel_size_): latice size of a single pixel.')
    st.markdown('**Temp Max boundary** (_temp_max_): set the boundary for the highest temperature.')
    st.markdown('**lower limit excluded area** (_cut_min_): A portion of the spectrum can be deleted before the analysis, here you set the lower limit of the range.')
    st.markdown('**upper limit excluded area** (_cut_max_): A portion of the spectrum can be deleted before the analysis, here you set the upper limit of the range.')
    st.markdown('**Geometrical Cross-section resolution** (_resolution_geometrical_cross_section_): number no longer used, but still present in the database in case of future use.')
    # st.markdown('**Geometrical Cross-section resolution** (_resolution_geometrical_cross_section_): number of point used to integrate the area of the geometrical cross section of the particle (the process time is strongly dependend from this number). Numbers smaller than 10 are not reccomanded.')
    st.markdown('**number of point for the course testing** (_num_loop1_): Number of temperature probed in the course test during the feedback learning process.')
    st.markdown('**Number of point for the fine testing** (_num_loop2_): Number of temperature probed in the fine testing during the feedback learning process.')
    st.markdown('**R2 threshold** (_threshold_r2_): set the threshold for the quality in the fit.')
    st.markdown('**selected scan** (_selected_scan_): this number range from **0** to **num of spectrum**, when it is 0 the reference in the ratio is set to be the less intense spectrum, when it is 1 the reference in the ration is set to be the second less intense spectrum, and so on.')
    st.markdown('**T reference** (_T_RT_): Temperature of the reference.')


    # ███    ███  █████  ██ ███    ██
    # ████  ████ ██   ██ ██ ████   ██
    # ██ ████ ██ ███████ ██ ██ ██  ██
    # ██  ██  ██ ██   ██ ██ ██  ██ ██
    # ██      ██ ██   ██ ██ ██   ████


if st.checkbox('table database'):
    st.subheader('file names')
    st.dataframe(df.T['PL_name'].tolist())
    st.subheader('Parameters')
    st.dataframe(df.T.reset_index(drop=True).drop(['index'], axis = 1))
    download_file(df, 'Parameters_table')

PL_name_temp = ''
PL_mane = st.sidebar.text_input('name file', PL_name_temp)

vet_param = [PL_mane, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'mat']

scelta = st.sidebar.radio('type of parameters',('load','new'))
if scelta == 'load':
    if PL_mane in df.loc['PL_name']:
        vet_param = [df[PL_mane]['PL_name'], df[PL_mane]['numb_of_spect'], df[PL_mane]['num_of_scan'], df[PL_mane]['bkg_position'],
                 df[PL_mane]['x_bkg'], df[PL_mane]['y_bkg'], df[PL_mane]['remove_cosmic_ray'], df[PL_mane]['derivative_threshold'],
                 df[PL_mane]['wave_inf'], df[PL_mane]['wave_sup'], df[PL_mane]['num_bin'], df[PL_mane]['radius'],
                 df[PL_mane]['laser_power'], df[PL_mane]['AS_min'], df[PL_mane]['AS_max'], df[PL_mane]['S_min'],
                 df[PL_mane]['S_max'], df[PL_mane]['row_y_start_delete'], df[PL_mane]['row_y_stop_delete'], df[PL_mane]['pixel_for_delete_x'],
                 df[PL_mane]['pixel_for_delte_y'], df[PL_mane]['laser_type'], df[PL_mane]['pixel_size'], df[PL_mane]['temp_max'],
                 df[PL_mane]['cut_min'], df[PL_mane]['cut_max'], df[PL_mane]['resolution_geometrical_cross_section'], df[PL_mane]['num_loop1'],
                 df[PL_mane]['num_loop2'], df[PL_mane]['threshold_r2'], df[PL_mane]['selected_scan'], df[PL_mane]['T_RT'], df[PL_mane]['material']]
        st.subheader('Selected Parameters')
        st.write(df[PL_mane].drop(['index', 'PL_name']))
    else:
        st.error('The selected name is not in the database')
elif scelta == 'new':
    vet_param = [PL_mane, 10, 20, 0, 0, 0, 1, 0, 646, 656, 10, 0, 0, 590, 616, 645, 699, 20, 20, 0, 0, 633, 0.05, 550, 0, 0, 10, 30, 10, 0.5, 0, 300, 'mat']

(PL_mane, numb_of_spect, num_of_scan, bkg_position, x_bkg, y_bkg, remove_cosmic_ray, soglia_derivata, wave_inf,
wave_sup, num_bin, raggio, laser_power, AS_min, AS_max, S_min, S_max, riga_y_start, riga_y_stop, pixel_0_x, pixel_0_y,
laser_type, lato_cella, temp_max, cut_min, cut_max, punti_cross_section_geometrica, num_loop1, num_loop2, soglia_r2, selected_scan, T_RT, material) = vet_param

# empty_save = st.sidebar.empty()
# if st.sidebar.button('Delete'):
#     if PL_mane in exp_names:
#         delete_database(df, PL_mane)
#         st.warning("succesfully deleted the parameters for "+PL_mane)
#     else:
#         st.error(PL_mane + " is not in the database therefore cannot be deleted")

type_of_analysis = ['home', 'single', 'learning on single', 'loop numerators', 'T averaging', 'Derivative threshold estimator', 'set all parameters', 'Compare multi particles', 'Compare log vs standard', 'Global analysis']
side_selection = st.sidebar.radio('Select the type of analysis', type_of_analysis)


# ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████
# ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██
# ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████
# ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██
# ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████
if side_selection != type_of_analysis[7] and side_selection != type_of_analysis[0]:
    if side_selection != type_of_analysis[4] and side_selection != type_of_analysis[8]:
        num_of_scan = int(st.sidebar.text_input('num of scan along a single axe', vet_param[2]))
    if side_selection != type_of_analysis[5]:
        numb_of_spect = int(st.sidebar.text_input('num of spectrum', vet_param[1]))

    if side_selection == type_of_analysis[1] or side_selection == type_of_analysis[2] or side_selection == type_of_analysis[3] or side_selection == type_of_analysis[6] or side_selection == type_of_analysis[9]:
        bkg_position = st.sidebar.selectbox('bkg', ('min','max'), index = vet_param[3])
        x_bkg = int(st.sidebar.text_input('X bkg pixel', vet_param[4]))
        y_bkg = int(st.sidebar.text_input('Y bkg pixel', vet_param[5]))
        remove_cosmic_ray = st.sidebar.selectbox('remove cosmic ray', ('yes','no'), index = vet_param[6])
        soglia_derivata = int(st.sidebar.text_input('derivative threshold', vet_param[7]))

        wave_inf = float(st.sidebar.text_input('lower limit to calculate the stokes on the heat map(nm)', vet_param[8]))
        wave_sup = float(st.sidebar.text_input('upper limit to calculate the stokes on the heat map(nm)', vet_param[9]))
        num_bin = int(st.sidebar.text_input('num bin', vet_param[10]))
        raggio = float(st.sidebar.text_input('radius (um)', vet_param[11]))
        laser_power = float(st.sidebar.text_input('laser power (mW)', vet_param[12]))
        AS_min = float(st.sidebar.text_input('AS min (nm)', vet_param[13]))
        AS_max = float(st.sidebar.text_input('AS max (nm)', vet_param[14]))
        S_min = float(st.sidebar.text_input('S min (nm)', vet_param[15]))
        S_max = float(st.sidebar.text_input('S max (nm)', vet_param[16]))

        riga_y_start = int(st.sidebar.text_input('Y start delete', vet_param[17]))
        riga_y_stop = int(st.sidebar.text_input('Y stop delete', vet_param[18]))
        pixel_0_x = int(st.sidebar.text_input('X pixel for delete', vet_param[19]))
        pixel_0_y = int(st.sidebar.text_input('Y pixel for delete', vet_param[20]))
        laser_type = int(st.sidebar.text_input('laser wavelength (nm)', vet_param[21]))
        lato_cella = float(st.sidebar.text_input('single pixel size (um)', vet_param[22]))
        temp_max = float(st.sidebar.text_input('Temp Max boundary (K)', vet_param[23]))
        cut_min = float(st.sidebar.text_input('lower limit excluded area (nm)', vet_param[24]))
        cut_max = float(st.sidebar.text_input('upper limit excluded area (nm)', vet_param[25]))
        # punti_cross_section_geometrica = int(st.sidebar.text_input('Geometrical Cross-section resolution', vet_param[26]))
        punti_cross_section_geometrica = vet_param[26] #removed from the analisys but still here in case of future use
        material = st.sidebar.text_input('Material', vet_param[32])

    if side_selection == type_of_analysis[3] or side_selection == type_of_analysis[2] or side_selection == type_of_analysis[6] or side_selection == type_of_analysis[9]:
        num_loop1 = int(st.sidebar.text_input('number of point for the course testing', vet_param[27]))
        num_loop2 = int(st.sidebar.text_input('number of point for the fine testing', vet_param[28]))

    if side_selection == type_of_analysis[3] or side_selection == type_of_analysis[4] or side_selection == type_of_analysis[6] or side_selection == type_of_analysis[8] or side_selection == type_of_analysis[9]:
        soglia_r2 = float(st.sidebar.text_input('R2 threshold', vet_param[29]))

    if side_selection == type_of_analysis[1] or side_selection == type_of_analysis[2] or side_selection == type_of_analysis[6]:
        selected_scan = int(st.sidebar.text_input('selected scan', vet_param[30]))

    if side_selection == type_of_analysis[1] or side_selection == type_of_analysis[6]:
        T_RT = float(st.sidebar.text_input('T reference (K)', vet_param[31]))


# if empty_save.button('Saving'):
#     vet_param = [PL_mane, numb_of_spect, num_of_scan, bkg_position, x_bkg, y_bkg, remove_cosmic_ray, soglia_derivata,
#                  wave_inf, wave_sup, num_bin, raggio, laser_power, AS_min, AS_max, S_min, S_max, riga_y_start, riga_y_stop,
#                  pixel_0_x, pixel_0_y, laser_type, lato_cella,
#                  temp_max, cut_min, cut_max, punti_cross_section_geometrica, num_loop1, num_loop2, soglia_r2, selected_scan, T_RT, material]
#     if PL_mane in exp_names:
#         save_database(df[PL_mane]['index'] - 1, vet_param)
#         st.success("succesfully uploaded the parameters on "+PL_mane)
#     else:
#         save_database(df.loc['index'][-1], vet_param)
#         st.success("succesfully saved the parameters on "+PL_mane)
#         save_database(df.loc['index'][-1], vet_param)


        # ███████ ████████  █████  ██████  ████████      █████  ███    ██  █████  ██      ██ ███████ ██    ██ ███████
        # ██         ██    ██   ██ ██   ██    ██        ██   ██ ████   ██ ██   ██ ██      ██ ██       ██  ██  ██
        # ███████    ██    ███████ ██████     ██        ███████ ██ ██  ██ ███████ ██      ██ ███████   ████   ███████
        #      ██    ██    ██   ██ ██   ██    ██        ██   ██ ██  ██ ██ ██   ██ ██      ██      ██    ██         ██
        # ███████    ██    ██   ██ ██   ██    ██        ██   ██ ██   ████ ██   ██ ███████ ██ ███████    ██    ███████


static_store = get_static_store()

file_PL = st.file_uploader("Upload dataset", type = ["csv", "txt"])
if file_PL:
    if side_selection != type_of_analysis[4] and side_selection != type_of_analysis[0] and side_selection != type_of_analysis[7] and side_selection != type_of_analysis[6] and side_selection != type_of_analysis[8]:

        if side_selection != type_of_analysis[5] and side_selection != type_of_analysis[9]:
            scelta2 = st.radio('type of parameters',('standard scale', 'standard direct', 'log scale', 'direct log'))
            log_name = ''
            if scelta2 == 'log scale':
                st.latex(r'''\ln{\left(\frac{I_1}{I_2}\right)} = \frac{1}{\lambda}\left[\frac{\hbar c}{k_b} \left(\frac{1}{T_2} - \frac{1}{T_1} \right)\right] + \frac{\hbar c}{k_b \lambda_{laser}}\left(\frac{1}{T_1} - \frac{1}{T_2} \right) +  \ln{\left(\frac{P_1}{P_2}\right)}''')
                log_scale = 1
                log_name = '_log'
            elif scelta2 == 'standard scale':
                st.latex(r'''\frac{I_1}{I_2} =  \frac{P_1}{P_2} \frac{\left(e^{\frac{\hbar c}{k_b T_2} \left(\frac{1}{\lambda} - \frac{1}{\lambda_{laser}}\right)} - 1\right)}{\left(e^{\frac{\hbar c}{k_b T_1} \left(\frac{1}{\lambda} - \frac{1}{\lambda_{laser}} \right)} - 1 \right)}''')
                log_scale = 0
                log_name = '_standard'
            elif scelta2 == 'direct log':
                st.latex(r'''T_1 = \frac{\frac{\hbar c}{k_b} \left( \frac{1}{\lambda_{laser}} - \frac{1}{\lambda} \right)}{ \ln{\frac{I_1}{I_2}} - \ln{\frac{P1}{P2}} + \frac{\hbar c}{k_b T_2} \left( \frac{1}{\lambda_{laser}} - \frac{1}{\lambda} \right)}''')
                log_scale = 2
                log_name = '_direct'
            elif scelta2 == 'standard direct':
                st.latex(r'''T_1 = \frac{\frac{\hbar c}{k_b} \left( \frac{1}{\lambda} - \frac{1}{\lambda_{laser}} \right)}{\ln{\left( \frac{P_1}{P_2} \frac{I_2}{I_1} \left( e^{ \frac{\hbar c}{k_b T_2} \left( \frac{1}{\lambda} - \frac{1}{\lambda_{laser}} \right)} - 1 \right) + 1 \right)}}''')
                log_scale = 3
                log_name = '_standard_direct'
        else:
            log_scale = 0

        if st.button("Start"):
            raw_data = data_storage(file_PL).load_map(num_of_scan)
            if side_selection == type_of_analysis[1] or side_selection == type_of_analysis[2] or side_selection == type_of_analysis[3]:
                misure = mp(raw_data, num_of_scan=num_of_scan, log_scale = log_scale,
                            bkg_position = bkg_position, soglia_derivata = soglia_derivata, remove_cosmic_ray = remove_cosmic_ray,
                            nome_file = PL_mane+log_name, x_bkg = x_bkg, y_bkg = y_bkg, material = material+log_name)
            if side_selection == type_of_analysis[1]:#single
                save = 'yes'
                _, save_matr, save_matr2 = misure.good_to_go(wave_inf = wave_inf, wave_sup =wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect, raggio = raggio, laser_power = laser_power,
                                  AS_min = AS_min, AS_max = AS_max, S_max = S_max, S_min = S_min, riga_y_start = riga_y_start, riga_y_stop =riga_y_stop, pixel_0 = [pixel_0_x, pixel_0_y], T_RT = T_RT, salva = save,
                                  laser_type = laser_type, selected_scan = selected_scan, lato_cella = lato_cella, temp_max = temp_max, cut_min = cut_min, cut_max = cut_max,
                                  punti_cross_section_geometrica = punti_cross_section_geometrica)

                st.write("File Download")
                download_file(save_matr, PL_mane+log_name+'_power_vs_temp_single')
                download_file(save_matr2, PL_mane+log_name+'_compare')

            if side_selection == type_of_analysis[2]:#learning on single
                save = 'no'
                empty_write2 = st.empty()
                empty_bar2 = st.empty()

                empty_write = st.empty()
                empty_bar = st.empty()

                empty_plot = st.empty()
                empty_text = st.empty()
                save_matr, _ = loop().T_background_calc(empty_bar, empty_bar2, empty_write, empty_write2,empty_plot,empty_text,
                                misure = misure, PL_mane = PL_mane+log_name, salva = 'no', num_loop1 = num_loop1, num_loop2 =num_loop2, punti_cross_section_geometrica = punti_cross_section_geometrica,
                                  wave_inf = wave_inf, wave_sup =wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect, raggio = raggio, laser_power = laser_power, AS_min = AS_min,
                                  AS_max = AS_max, S_max = S_max, S_min = S_min, lato_cella = lato_cella, temp_max = temp_max, cut_min = cut_min, cut_max = cut_max,
                                  riga_y_start = riga_y_start, riga_y_stop =riga_y_stop, pixel_0 = [pixel_0_x,pixel_0_y], laser_type = laser_type, selected_scan = selected_scan)
                def retta(x, p0, p1):
                    return p0*x + p1
                x1 = np.array(save_matr['xx0'], dtype="float")
                y2 = np.array(save_matr['yy0'], dtype="float")
                par1, par2 = curve_fit(retta, x1, y2)
                y_fit2 = retta(x1, par1[0], par1[1])

                m = par1[0]
                q = par1[1]
                residual = y2 - y_fit2
                ss_res = np.sum(residual**2)
                ss_tot = np.sum((y2 - np.mean(y2))**2)
                if ss_tot == 0:
                    ss_tot = 1
                    ss_res = 1
                r2 = 1- (ss_res/ss_tot)

                p = len(par1)
                n = len(x1)
                alpha = 0.05 #95% confidence interval
                dof = max(0, len(x1) - len(par1)) #degree of freedom
                tval = tstud.ppf(1.0 - alpha/2., dof) #t-student value for the dof and confidence level
                sigma = np.diag(par2)**0.5
                m_err = sigma[0]*tval
                q_err = sigma[1]*tval

                y_fit2_up = retta(x1, m+(m_err/2), q+(q_err/2))
                y_fit2_down = retta(x1, m-(m_err/2), q-(q_err/2))

                ds().nuova_fig(indice_fig=15)
                ds().titoli(xtag='I [mW/um^2]', ytag = 'T [k]', titolo='')
                ds().dati(save_matr['xx0'], save_matr['yy0'], x_error = [save_matr['ex10'], save_matr['ex20']], y_error = save_matr['ey0'], scat_plot = 'err')
                ds().dati(x1, y_fit2, colore='black', descrizione=str(round(par1[0],2)) + '*X + ' + str(round(par1[1],2))+'\n'+
                          str(round(m,2))+' +/- '+ str(round(m_err,2))+'\n'+str(round(q,2))+' +/- '+ str(round(q_err,2)))
                plt.fill_between(x1, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
                ds().legenda()
                st.pyplot()

                st.write("File Download")
                download_file(save_matr, PL_mane+log_name+'_power_vs_temp_single_after_learning')

            if side_selection == type_of_analysis[3]:#loop numerator
                save = 'no'
                empty_top_write = st.empty()
                empty_top_bar = st.empty()

                empty_write2 = st.empty()
                empty_bar2 = st.empty()

                empty_write = st.empty()
                empty_bar = st.empty()

                empty_plot = st.empty()
                empty_text = st.empty()
                save_matr,matr_x_compare = loop().switch_numeratore(empty_top_write, empty_top_bar, empty_bar, empty_bar2, empty_write, empty_write2,empty_plot,empty_text,
                                misure = misure, PL_mane = PL_mane+log_name, salva = 'no', num_loop1 = num_loop1, num_loop2 =num_loop2, punti_cross_section_geometrica = punti_cross_section_geometrica,
                                  wave_inf = wave_inf, wave_sup =wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect, raggio = raggio, laser_power = laser_power, AS_min = AS_min,
                                  AS_max = AS_max, S_max = S_max, S_min = S_min, lato_cella = lato_cella, temp_max = temp_max, cut_min = cut_min, cut_max = cut_max,
                                  riga_y_start = riga_y_start, riga_y_stop =riga_y_stop, pixel_0 = [pixel_0_x,pixel_0_y], laser_type = laser_type)
                matr_powers_quality_selected, average_power_temp, average_power_temp_quality_selected, m0 = loop().T_averageing(data = save_matr, numb_of_spect=numb_of_spect, soglia_r2 = soglia_r2)
                st.write("File Download")
                download_file(save_matr, PL_mane+log_name+'_power_vs_temp_over_loop')
                download_file(matr_powers_quality_selected, PL_mane+log_name+'_powers_quality_selected')
                download_file(average_power_temp, PL_mane+log_name+'_average_power_vs_temp')
                download_file(average_power_temp_quality_selected, PL_mane+log_name+'_average_power_vs_temp_quality_selected')
                save_matr2 = pd.DataFrame()
                save_matr2['signal_quality'] = matr_x_compare['signal_quality']
                save_matr2['signal_speed'] = m0
                save_matr2['radius'] = matr_x_compare['radius']
                save_matr2['laser_type'] = matr_x_compare['laser_type']
                save_matr2['laser_power'] = matr_x_compare['laser_power']
                save_matr2['material'] = matr_x_compare['material']
                download_file(save_matr2, PL_mane+log_name+'_compare')

            if side_selection == type_of_analysis[9]:#global analysis
                save = 'no'
                st.subheader('type of analysis')
                empty_type_bar = st.empty()

                empty_top_write = st.empty()
                empty_top_bar = st.empty()

                empty_write2 = st.empty()
                empty_bar2 = st.empty()

                empty_write = st.empty()
                empty_bar = st.empty()

                empty_plot = st.empty()
                empty_text = st.empty()

                log_name_vet = ['_standard', '_standard_direct', '_log', '_direct']
                log_scale_vet = [0, 3, 1, 2]

                my_bar0 = empty_type_bar.progress(0)
                dict_save_matr = dict()
                dict_save_matr2 = dict()
                for i in range(len(log_scale_vet)):

                    perc_progr = round(i*(100/3))
                    my_bar0.progress(perc_progr)

                    log_name = log_name_vet[i]
                    log_scale = log_scale_vet[i]

                    misure = mp(raw_data, num_of_scan=num_of_scan, log_scale = log_scale,
                                bkg_position = bkg_position, soglia_derivata = soglia_derivata, remove_cosmic_ray = remove_cosmic_ray,
                                nome_file = PL_mane+log_name, x_bkg = x_bkg, y_bkg = y_bkg, material = material+log_name)

                    save_matr, matr_x_compare = loop().switch_numeratore(empty_top_write, empty_top_bar, empty_bar, empty_bar2, empty_write, empty_write2,empty_plot,empty_text,
                                    misure = misure, PL_mane = PL_mane+log_name, salva = 'no', num_loop1 = num_loop1, num_loop2 =num_loop2, punti_cross_section_geometrica = punti_cross_section_geometrica,
                                      wave_inf = wave_inf, wave_sup =wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect, raggio = raggio, laser_power = laser_power, AS_min = AS_min,
                                      AS_max = AS_max, S_max = S_max, S_min = S_min, lato_cella = lato_cella, temp_max = temp_max, cut_min = cut_min, cut_max = cut_max,
                                      riga_y_start = riga_y_start, riga_y_stop =riga_y_stop, pixel_0 = [pixel_0_x,pixel_0_y], laser_type = laser_type)

                    dict_save_matr[log_name] = save_matr
                    dict_save_matr2[log_name] = matr_x_compare

                    st.subheader('Plot'+log_name)
                    matr_powers_quality_selected, average_power_temp, average_power_temp_quality_selected, m0 = loop().T_averageing(data = save_matr, numb_of_spect=numb_of_spect, soglia_r2 = soglia_r2, on_plot2 = False)

                    st.write("File Download")
                    download_file(dict_save_matr[log_name], PL_mane+log_name+'_power_vs_temp_over_loop')

                    save_matr2 = pd.DataFrame()
                    save_matr2['signal_quality'] = dict_save_matr2[log_name]['signal_quality']
                    save_matr2['signal_speed'] = m0
                    save_matr2['radius'] = dict_save_matr2[log_name]['radius']
                    save_matr2['laser_type'] = dict_save_matr2[log_name]['laser_type']
                    save_matr2['laser_power'] = dict_save_matr2[log_name]['laser_power']
                    save_matr2['material'] = dict_save_matr2[log_name]['material']
                    download_file(save_matr2, PL_mane+log_name+'_compare')

                my_bar0.progress(100)

            if side_selection == type_of_analysis[5]:#derivative threshold estimator
                preprocess(raw_data, num_of_scan).calc_deriv_thershold()

    elif side_selection == type_of_analysis[4]:#T averaging
        st.write('It requires "_power_vs_temp_over_loop" file generated by "loop numerators" or "Global analisys"')
        if st.button("Start"):
            file_PL = pd.DataFrame(file_PL)
            file_PL[0] = file_PL[0].str.replace('\r\n','')
            file_PL = file_PL[0].str.split(",", expand=True)
            colonne = file_PL.iloc[0].tolist()
            file_PL.columns = colonne
            file_PL = file_PL.drop([0], axis=0)
            file_PL = file_PL.apply(pd.to_numeric, downcast='float')

            matr_powers_quality_selected, average_power_temp, average_power_temp_quality_selected, _ = loop().T_averageing(data = file_PL, numb_of_spect=numb_of_spect, soglia_r2 = soglia_r2)
            st.write("File Download")
            download_file(matr_powers_quality_selected, PL_mane+'_powers_quality_selected')
            download_file(average_power_temp, PL_mane+'_average_power_vs_temp')
            download_file(average_power_temp_quality_selected, PL_mane+'_average_power_vs_temp_quality_selected')

    elif side_selection == type_of_analysis[8]:#compare log vs standard
        file_PL2 = st.file_uploader("Upload second dataset (red line)", type = ["csv", "txt"])
        st.write('It requires "_power_vs_temp_over_loop" file generated by "loop numerators" or "Global analisys"')
        if st.button("Start"):
            file_PL = pd.DataFrame(file_PL)
            file_PL[0] = file_PL[0].str.replace('\r\n','')
            file_PL = file_PL[0].str.split(",", expand=True)
            colonne = file_PL.iloc[0].tolist()
            file_PL.columns = colonne
            file_PL = file_PL.drop([0], axis=0)
            file_PL = file_PL.apply(pd.to_numeric, downcast='float')

            file_PL2 = pd.DataFrame(file_PL2)
            file_PL2[0] = file_PL2[0].str.replace('\r\n','')
            file_PL2 = file_PL2[0].str.split(",", expand=True)
            colonne = file_PL2.iloc[0].tolist()
            file_PL2.columns = colonne
            file_PL2 = file_PL2.drop([0], axis=0)
            file_PL2 = file_PL2.apply(pd.to_numeric, downcast='float')

            T_quality_selected, _, average_T_quality_selected, _ = loop().T_averageing(data = file_PL, numb_of_spect=numb_of_spect, soglia_r2 = soglia_r2, on_plot = False, on_plot2 = False)
            T_quality_selected2, _, average_T_quality_selected2, _ = loop().T_averageing(data = file_PL2, numb_of_spect=numb_of_spect, soglia_r2 = soglia_r2, on_plot = False, on_plot2 = False)
            loop().log_standard_plot(file_PL, T_quality_selected, average_T_quality_selected,
                                     file_PL2, T_quality_selected2, average_T_quality_selected2, numb_of_spect)

else:
    static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page

if side_selection == type_of_analysis[7]:#compare muilti particles
    data_concat = pd.DataFrame()
    uploadfile = st.file_uploader('Load zip file here as alternative to multi-drop', 'zip')
    st.write('It requires "_compare" file generated by "single" or "loop numerator" or "Global analysis"')
    if uploadfile or file_PL:
        if uploadfile:
            zf = zipfile.ZipFile(uploadfile)
            files = dict()
            for i, name in enumerate(zf.namelist()):
                files[i] = pd.read_csv(zf.open(name))
                files[i]['radius_nm'] = files[i]['radius']*1000
                files[i]['normalized_signal_quality'] = (files[i]['signal_quality']/files[i]['laser_power'])
                # st.write(files[i])
                data_concat = pd.concat([data_concat, files[i]], ignore_index = False, axis=0)
            # st.write(data_concat)
        else:
            if file_PL:
                value = file_PL.getvalue()
                data_compare = pd.DataFrame(file_PL)
                data_compare[0] = data_compare[0].str.replace('\r\n','')
                data_compare = data_compare[0].str.split(",", expand=True)
                colonne = data_compare.iloc[0].tolist()
                data_compare.columns = colonne
                data_compare = data_compare.drop([0], axis=0)
                mod_type = ['signal_quality', 'signal_speed', 'radius', 'laser_type', 'laser_power']
                data_compare[mod_type] = data_compare[mod_type].astype(float)

                if not value in static_store:
                    static_store[file_PL] = data_compare

                for value in static_store:
                    static_store[value]['radius_nm'] = static_store[value]['radius']*1000
                    static_store[value]['normalized_signal_quality'] = (static_store[value]['signal_quality']/static_store[value]['laser_power'])
                    data_concat = pd.concat([data_concat, static_store[value]], ignore_index = False, axis=0)

        data_concat = data_concat.sort_values(by = ['radius_nm'])
        st.table(data_concat)

        loop().bar_plot(data_concat)

        if st.button("Clear file list"):
            static_store.clear()
