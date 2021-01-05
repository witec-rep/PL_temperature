import numpy as np
from AJ_draw import disegna as ds
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats.distributions import t as tstud

palette = ['#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']

class loop:

    # ████████     ██████   ██████   ██████  ███    ███      ██████  █████  ██       ██████
    #    ██        ██   ██ ██    ██ ██    ██ ████  ████     ██      ██   ██ ██      ██
    #    ██        ██████  ██    ██ ██    ██ ██ ████ ██     ██      ███████ ██      ██
    #    ██        ██   ██ ██    ██ ██    ██ ██  ██  ██     ██      ██   ██ ██      ██
    #    ██        ██   ██  ██████   ██████  ██      ██      ██████ ██   ██ ███████  ██████



    def T_background_calc(self, empty_bar, empty_bar2, empty_write, empty_write2,empty_plot,empty_text, punti_cross_section_geometrica,
                           misure, temp_max, PL_mane, wave_inf, wave_sup, num_bin,  numb_of_spect,  raggio, laser_power,  riga_y_start, riga_y_stop, pixel_0, laser_type, salva,  S_min = 645, S_max= 699,
                          AS_min = 590, AS_max = 616, lato_cella = 0.05, cut_min = 0, cut_max = 0, selected_scan = 0, num_loop1 = 30, num_loop2 = 10):

        indice = np.linspace(295, temp_max, num_loop1)
        T_matr = np.zeros((len(indice),3))
        T_matr = T_matr+10
        j = 0
        empty_write.subheader('Course Tuning Reference Temperature')
        my_bar = empty_bar.progress(0)
        empty_write2.subheader('Fine Tuning Reference Temperature')
        my_bar2 = empty_bar2.progress(0)
        for i in indice:
            T_RT,_,_ = misure.good_to_go(wave_inf = wave_inf, wave_sup = wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect,  raggio = raggio, temp_max = temp_max,
                                     S_min=S_min, S_max = S_max, AS_min = AS_min, AS_max = AS_max, laser_power = laser_power,  riga_y_start = riga_y_start, riga_y_stop =riga_y_stop,
                                     pixel_0 = pixel_0, laser_type = laser_type, T_RT = i, salva = salva, lato_cella = lato_cella, cut_min = cut_min, cut_max = cut_max, selected_scan = selected_scan,
                                     punti_cross_section_geometrica = punti_cross_section_geometrica)
            T_matr[j,0] = T_RT #room temperature in K
            T_matr[j,1] = T_RT - 295#room temperature in C
            T_matr[j,2] = i# reference temperature
            ds().nuova_fig(10)
            ds().titoli(titolo='learning plot', xtag='T reference', ytag="Room Temperature")
            ds().dati([0,1000], [295,295], colore='red')
            ds().dati([295,295], [0,1000], colore='green')
            ds().dati(x = T_matr[:,2], y = T_matr[:,0], scat_plot ='scat', larghezza_riga = 15)
            ds().range_plot(bottomX = 220, topX = temp_max, bottomY = 220, topY = temp_max)
            empty_plot.pyplot()
            if T_matr[j,1] > 0:
                break
            j = j + 1
            perc_progr = round(j*(100/len(indice)))
            my_bar.progress(perc_progr)

        #trova la temp piu bassa e parte da li per cercare il punto piu vicino alla room temperature
        min_abs = T_matr[0,1]
        imin_abs = 0
        for i in range(len(T_matr[:,1])):
            if T_matr[i,1]<min_abs:
                min_abs = T_matr[i,1]
                imin_abs = i

        if min_abs < (220 - 295):
            min_abs = T_matr[0,1]
            imin_abs = 0

        # i valori non usati che erano inizialmente settati a 10 vengono risettati a -9999
        if j<len(T_matr[:,1]):
            for i in range(j+1,len(T_matr[:,1])):
                T_matr[i,1] = -9999

        min = np.sqrt(min_abs*min_abs)
        imin = imin_abs
        for i in range(imin_abs, len(indice)):
            if np.sqrt(T_matr[i,1]*T_matr[i,1])<min:
                min = np.sqrt(T_matr[i,1]*T_matr[i,1])
                imin = i

        delta_T = T_matr[1,2] - T_matr[0,2]
        indice = np.linspace(T_matr[imin,2]-delta_T, T_matr[imin,2]+delta_T, num_loop2)

        T_matr2 = np.zeros((len(indice),3))
        j = 0

        for i in indice:
            T_RT,_,_ = misure.good_to_go(wave_inf = wave_inf, wave_sup = wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect,  raggio = raggio, temp_max = temp_max,
                                     S_min=S_min, S_max = S_max, AS_min = AS_min, AS_max = AS_max, laser_power = laser_power,  riga_y_start = riga_y_start, riga_y_stop =riga_y_stop,
                                     pixel_0 = pixel_0, laser_type = laser_type, T_RT = i, salva = salva, lato_cella = lato_cella, cut_min = cut_min, cut_max = cut_max, selected_scan = selected_scan,
                                     punti_cross_section_geometrica = punti_cross_section_geometrica)
            T_matr2[j,0] = T_RT
            T_matr2[j,1] = T_RT - 295
            T_matr2[j,2] = i
            j = j + 1
            perc_progr = round(j*(100/len(indice)))
            my_bar2.progress(perc_progr)
            ds().nuova_fig(10)
            ds().titoli(titolo='learning plot', xtag='T reference', ytag="Room Temperature")
            ds().dati([0,1000], [295,295], colore='red')
            ds().dati([295,295], [0,1000], colore='green')
            ds().dati(x = T_matr[:,2], y = T_matr[:,0], scat_plot ='scat', larghezza_riga = 15)
            ds().dati(x = T_matr2[:,2], y = T_matr2[:,0], scat_plot ='scat', larghezza_riga = 15, colore=palette[0])
            ds().range_plot(bottomX = 220, topX = temp_max, bottomY = 220, topY = temp_max)
            empty_plot.pyplot()
        min = np.sqrt(T_matr2[0,1]*T_matr2[0,1])
        imin = 0
        for i in range(len(indice)):
            if np.sqrt(T_matr2[i,1]*T_matr2[i,1])<min:
                min = np.sqrt(T_matr2[i,1]*T_matr2[i,1])
                imin = i
        empty_text.subheader('Temperature of the Reference: '+str(round(T_matr2[imin,2],2))+' [K]')

        _,matr, matr_x_compare = misure.good_to_go(wave_inf = wave_inf, wave_sup = wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect,  raggio = raggio, temp_max = temp_max,
                                 S_min=S_min, S_max = S_max, AS_min = AS_min, AS_max = AS_max, laser_power = laser_power,  riga_y_start = riga_y_start, riga_y_stop =riga_y_stop,
                                 pixel_0 = pixel_0, laser_type = laser_type, T_RT = T_matr2[imin,2], salva = salva, lato_cella = lato_cella, cut_min = cut_min, cut_max = cut_max, selected_scan = selected_scan,
                                 punti_cross_section_geometrica = punti_cross_section_geometrica)

        data = pd.DataFrame()
        data['xx'+str(selected_scan)] = matr['average_intensity_new'].tolist()
        data['yy'+str(selected_scan)] = matr['Temperature'].tolist()
        data['ex1'+str(selected_scan)] = matr['average_intensity_err_low'].tolist()
        data['ex2'+str(selected_scan)] = matr['average_intensity_err_up'].tolist()
        data['ey'+str(selected_scan)] = matr['Temperature_err'].tolist()
        data['r2'+str(selected_scan)] = matr['Temperature_r2'].tolist()
        return data, matr_x_compare


        # ██       ██████   ██████  ██████      ███    ██ ██    ██ ███    ███ ███████ ██████   █████  ████████  ██████  ██████
        # ██      ██    ██ ██    ██ ██   ██     ████   ██ ██    ██ ████  ████ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
        # ██      ██    ██ ██    ██ ██████      ██ ██  ██ ██    ██ ██ ████ ██ █████   ██████  ███████    ██    ██    ██ ██████
        # ██      ██    ██ ██    ██ ██          ██  ██ ██ ██    ██ ██  ██  ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
        # ███████  ██████   ██████  ██          ██   ████  ██████  ██      ██ ███████ ██   ██ ██   ██    ██     ██████  ██   ██




    def switch_numeratore(self, empty_top_write, empty_top_bar, empty_bar, empty_bar2, empty_write, empty_write2,empty_plot,empty_text, temp_max, num_loop1, num_loop2, punti_cross_section_geometrica,
                          misure, PL_mane, wave_inf, wave_sup, num_bin,  numb_of_spect,  raggio, laser_power,  riga_y_start, riga_y_stop, pixel_0, laser_type, salva,  S_min = 645, S_max= 699,
                          AS_min = 590, AS_max = 616, lato_cella = 0.05, cut_min = 0, cut_max = 0):
        empty_top_write.subheader('Loop over all numerators')
        my_bar0 = empty_top_bar.progress(0)
        data = pd.DataFrame()
        for i in range(numb_of_spect):
            data_row, matr_x_compare = self.T_background_calc( empty_bar, empty_bar2, empty_write, empty_write2,empty_plot,empty_text,
                                misure = misure, PL_mane = PL_mane, salva = 'no', num_loop1 = num_loop1, num_loop2 = num_loop2,
                                AS_max = AS_max, S_max = S_max, S_min = S_min, lato_cella = lato_cella, temp_max = temp_max, cut_min = cut_min, cut_max = cut_max,
                              wave_inf = wave_inf, wave_sup =wave_sup, num_bin = num_bin,  numb_of_spect = numb_of_spect, raggio = raggio, laser_power = laser_power, AS_min = AS_min,
                              riga_y_start = riga_y_start, riga_y_stop =riga_y_stop, pixel_0 = pixel_0, laser_type = laser_type, selected_scan = i,
                              punti_cross_section_geometrica = punti_cross_section_geometrica)
            data = pd.concat([data, data_row], ignore_index = False, axis=1)
            perc_progr = round(i*(100/numb_of_spect))
            my_bar0.progress(perc_progr + 1)
        my_bar0.progress(100)

        return data, matr_x_compare


        #  ██████  ██████  ███    ██ ███████ ██ ██████  ███████ ███    ██  ██████ ███████     ██ ███    ██ ████████ ███████ ██████  ██    ██  █████  ██
        # ██      ██    ██ ████   ██ ██      ██ ██   ██ ██      ████   ██ ██      ██          ██ ████   ██    ██    ██      ██   ██ ██    ██ ██   ██ ██
        # ██      ██    ██ ██ ██  ██ █████   ██ ██   ██ █████   ██ ██  ██ ██      █████       ██ ██ ██  ██    ██    █████   ██████  ██    ██ ███████ ██
        # ██      ██    ██ ██  ██ ██ ██      ██ ██   ██ ██      ██  ██ ██ ██      ██          ██ ██  ██ ██    ██    ██      ██   ██  ██  ██  ██   ██ ██
        #  ██████  ██████  ██   ████ ██      ██ ██████  ███████ ██   ████  ██████ ███████     ██ ██   ████    ██    ███████ ██   ██   ████   ██   ██ ███████


    def interval_confidence(self, x_tot, y_tot):
        def retta(x, p0, p1):
            return p0*x + p1

        par1, par2 = curve_fit(retta, x_tot, y_tot)
        y_fit2 = retta(x_tot, par1[0], par1[1])

        m = par1[0]
        q = par1[1]
        # residual = y_tot - y_fit2
        # ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_tot - np.mean(y_tot))**2)
        if ss_tot == 0:
            ss_tot = 1
            # ss_res = 1
        # r2 = 1- (ss_res/ss_tot)
        # p = len(par1)
        # n = len(x_tot)
        alpha = 0.05 #95% confidence interval
        dof = max(0, len(x_tot) - len(par1)) #degree of freedom
        tval = tstud.ppf(1.0 - alpha/2., dof) #t-student value for the dof and confidence level
        sigma = np.diag(par2)**0.5
        m_err = sigma[0]*tval
        q_err = sigma[1]*tval

        y_fit2_up = retta(x_tot, m+(m_err/2), q+(q_err/2))
        y_fit2_down = retta(x_tot, m-(m_err/2), q-(q_err/2))

        return y_fit2, m, m_err, q, q_err, y_fit2_up, y_fit2_down


        # ████████      █████  ██    ██ ███████ ██████   █████   ██████  ███████
        #    ██        ██   ██ ██    ██ ██      ██   ██ ██   ██ ██       ██
        #    ██        ███████ ██    ██ █████   ██████  ███████ ██   ███ █████
        #    ██        ██   ██  ██  ██  ██      ██   ██ ██   ██ ██    ██ ██
        #    ██        ██   ██   ████   ███████ ██   ██ ██   ██  ██████  ███████


    def T_averageing(self, data, numb_of_spect, soglia_r2 = 0.9, on_plot = True, on_plot2 = True):
        df_temp = pd.DataFrame()
        data_r2 = pd.DataFrame()
        data_T = pd.DataFrame()
        for i in range(data.shape[0]):
            data_r2 = pd.concat((data_r2, data['r2'+str(i)]), ignore_index = 0, axis= 1)
            data_T = pd.concat((data_T, data['yy'+str(i)]), ignore_index = 0, axis= 1)

        for j in range(numb_of_spect):
            temp_i = []
            df_temp_temp = pd.DataFrame()

            for i in range(numb_of_spect):
                if data['r2'+str(i)].iloc[j] > soglia_r2 or data['r2'+str(i)].iloc[j] == 0:
                    temp_i.append(data['yy'+str(i)].iloc[j])

            df_temp_temp['P'+str(j)] = temp_i
            df_temp = pd.concat([df_temp, df_temp_temp], ignore_index = True, axis=1)

        list_col_df = ['P'+str(i) for i in range(numb_of_spect)]
        df_temp.columns = list_col_df
        temp_media_vet = np.array(df_temp[list_col_df].mean(axis = 0), dtype="float")
        temp_sigma_vet = np.array(df_temp[list_col_df].std(axis = 0).fillna(0), dtype="float")

#################################################################################################################
        x_err = np.zeros((2, numb_of_spect))
        x_err[0][:] = data['ex10']
        x_err[1][:] = data['ex20']
        list_col = ['yy'+str(i) for i in range(numb_of_spect)]
        list_col_x = ['xx'+str(i) for i in range(numb_of_spect)]

        x_tot = pd.DataFrame()
        y_tot = pd.DataFrame()
        for i in range(len(list_col_x)):
            x_tot = pd.concat([x_tot, data[list_col_x[i]]])
            y_tot = pd.concat([y_tot, data[list_col[i]]])

        x_tot = x_tot[0].to_numpy()
        y_tot = y_tot[0].to_numpy()

        y_fit2, m0, m_err, q, q_err, y_fit2_up, y_fit2_down = self.interval_confidence(x_tot, y_tot)

        if on_plot == True:
            ds().nuova_fig(20)
            ds().titoli(xtag='P [uW]', ytag = 'T [k]', titolo='without quality filter')
            for i in range(numb_of_spect):
                ds().dati(x = data['xx0'], y = data['yy'+str(i)], colore=palette[i], scat_plot ='scat', larghezza_riga =15)
            ds().dati(x = data['xx0'], y = data[list_col].mean(axis = 1), colore='black', scat_plot = 'scat', larghezza_riga =12)
            ds().dati(x = data['xx0'], y = data[list_col].mean(axis = 1), x_error = x_err, y_error= data[list_col].std(axis = 1), colore='black', scat_plot = 'err')
            ds().dati(x_tot, y_fit2, colore='black', descrizione='Y = '+str(round(m0,2)) + '*X + ' + str(round(q,2))+'\n'+
                      'm = '+str(round(m0,2))+' +/- '+ str(round(m_err,2))+'\n q = '+str(round(q,2))+' +/- '+ str(round(q_err,2)))
            plt.fill_between(x_tot, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
            ds().legenda()
            st.pyplot()
#################################################################################################################

#################################################################################################################
        df_temp_tot = df_temp.T
        x_tot = pd.DataFrame()
        y_tot = pd.DataFrame()
        df_tot = pd.DataFrame()

        for col in df_temp_tot.columns:
            y_tot = pd.concat([y_tot, df_temp_tot[col]])
            x_tot = pd.concat([x_tot, data['xx0']])

        df_tot['x'] = x_tot[0]
        df_tot.reset_index(inplace = True)
        df_tot.drop(['index'], axis=1, inplace=True)
        y_tot.reset_index(inplace = True)
        y_tot.drop(['index'], axis=1, inplace=True)
        df_tot['y'] = y_tot[0]
        df_tot.dropna(inplace = True)

        x_tot = df_tot['x'].to_numpy()
        y_tot = df_tot['y'].to_numpy()

        y_fit2, m, m_err, q, q_err, y_fit2_up, y_fit2_down = self.interval_confidence(x_tot, y_tot)

        if on_plot2 == True:
            ds().nuova_fig(21)
            ds().titoli(xtag='P [uW]', ytag = 'T [k]', titolo='with quality filter')
            for i in range(df_temp.shape[0]):
                ds().dati(x = data['xx0'], y = df_temp.iloc[i], colore=palette[i], scat_plot ='scat', larghezza_riga =15)
            ds().dati(x = data['xx0'], y = temp_media_vet, colore='black', scat_plot = 'scat', larghezza_riga =12)
            ds().dati(x = data['xx0'], y = temp_media_vet, x_error = x_err, y_error= temp_sigma_vet, colore='black', scat_plot = 'err')
            ds().dati(x_tot, y_fit2, colore='black', descrizione='Y = '+str(round(m,2)) + '*X + ' + str(round(q,2))+'\n'+
                      'm = '+str(round(m,2))+' +/- '+ str(round(m_err,2))+'\n q = '+str(round(q,2))+' +/- '+ str(round(q_err,2)))
            plt.fill_between(x_tot, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
            ds().legenda()
            st.pyplot()
#################################################################################################################

            st.subheader('Temperature matrix')
            st.write(data_T)
            st.subheader('Quality matrix')
            st.write(data_r2)
            st.subheader('Temperature matrix after selection')
            st.write(df_temp.T)

        average_power_temp = pd.DataFrame()
        average_power_temp['power'] = data['xx0']
        average_power_temp['Temp'] = data[list_col].mean(axis = 1)
        average_power_temp['Err power 1'] = data['ex10']
        average_power_temp['Err power 2'] = data['ex20']
        average_power_temp['Err Temp'] = data[list_col].std(axis = 1)

        average_power_temp_quality_selected = pd.DataFrame()
        average_power_temp_quality_selected['power'] = data['xx0']
        average_power_temp_quality_selected['Temp'] = temp_media_vet
        average_power_temp_quality_selected['Err power 1'] = data['ex10']
        average_power_temp_quality_selected['Err power 2'] = data['ex20']
        average_power_temp_quality_selected['Err Temp'] = temp_sigma_vet

        return df_temp, average_power_temp, average_power_temp_quality_selected, m0



        #  ██████  ██████  ███    ███ ██████   █████  ██████  ███████ ███████  ██████  ███    ██     ██████      ██      ██ ███    ██ ███████     ██████  ██       ██████  ████████
        # ██      ██    ██ ████  ████ ██   ██ ██   ██ ██   ██ ██      ██      ██    ██ ████   ██          ██     ██      ██ ████   ██ ██          ██   ██ ██      ██    ██    ██
        # ██      ██    ██ ██ ████ ██ ██████  ███████ ██████  █████   ███████ ██    ██ ██ ██  ██      █████      ██      ██ ██ ██  ██ █████       ██████  ██      ██    ██    ██
        # ██      ██    ██ ██  ██  ██ ██      ██   ██ ██   ██ ██           ██ ██    ██ ██  ██ ██     ██          ██      ██ ██  ██ ██ ██          ██      ██      ██    ██    ██
        #  ██████  ██████  ██      ██ ██      ██   ██ ██   ██ ███████ ███████  ██████  ██   ████     ███████     ███████ ██ ██   ████ ███████     ██      ███████  ██████     ██


    def log_standard_plot(self, T, T_quality_selected, average_T_quality_selected,
                          T2, T_quality_selected2, average_T_quality_selected2, numb_of_spect):

        def plot1(T, colore):
            list_col = ['yy'+str(i) for i in range(numb_of_spect)]
            list_col_x = ['xx'+str(i) for i in range(numb_of_spect)]
            x_err = np.zeros((2, numb_of_spect))
            x_err[0][:] = T['ex10']
            x_err[1][:] = T['ex20']

            x_tot = pd.DataFrame()
            y_tot = pd.DataFrame()
            for i in range(len(list_col)):
                x_tot = pd.concat([x_tot, T[list_col_x[i]]])
                y_tot = pd.concat([y_tot, T[list_col[i]]])

            x_tot = x_tot[0].to_numpy()
            y_tot = y_tot[0].to_numpy()

            y_fit2, m, m_err, q, q_err, y_fit2_up, y_fit2_down = self.interval_confidence(x_tot, y_tot)

            ds().nuova_fig(40)
            ds().titoli(xtag='P [uW]', ytag = 'T [k]', titolo='without quality filter')
            for i in range(numb_of_spect):
                ds().dati(x = T['xx0'], y = T['yy'+str(i)], colore=palette[i], scat_plot ='scat', larghezza_riga =15)
            ds().dati(x = T['xx0'], y = T[list_col].mean(axis = 1), colore=colore, scat_plot = 'scat', larghezza_riga =12)
            ds().dati(x = T['xx0'], y = T[list_col].mean(axis = 1), x_error = x_err, y_error= T[list_col].std(axis = 1), colore=colore, scat_plot = 'err')
            ds().dati(x_tot, y_fit2, colore=colore, descrizione='Y = '+str(round(m,2)) + '*X + ' + str(round(q,2))+'\n'+
                      'm = '+str(round(m,2))+' +/- '+ str(round(m_err,2))+'\n q = '+str(round(q,2))+' +/- '+ str(round(q_err,2)))
            plt.fill_between(x_tot, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
            ds().legenda()

        plot1(T, 'blue')
        plot1(T2, 'red')
        ds().nuova_fig(40)
        st.pyplot()

        def plot2(T, T_quality_selected, average_T_quality_selected, colore):

            df_temp_tot = T_quality_selected.T
            x_tot = pd.DataFrame()
            y_tot = pd.DataFrame()
            df_tot = pd.DataFrame()

            for col in df_temp_tot.columns:
                y_tot = pd.concat([y_tot, df_temp_tot[col]])
                x_tot = pd.concat([x_tot, T['xx0']])

            df_tot['x'] = x_tot[0]
            df_tot.reset_index(inplace = True)
            df_tot.drop(['index'], axis=1, inplace=True)
            y_tot.reset_index(inplace = True)
            y_tot.drop(['index'], axis=1, inplace=True)
            df_tot['y'] = y_tot[0]
            df_tot.dropna(inplace = True)

            x_tot = df_tot['x'].to_numpy()
            y_tot = df_tot['y'].to_numpy()

            x_err = np.zeros((2, numb_of_spect))
            x_err[0][:] = T['ex10']
            x_err[1][:] = T['ex20']

            y_fit2, m, m_err, q, q_err, y_fit2_up, y_fit2_down = self.interval_confidence(x_tot, y_tot)

            ds().nuova_fig(41)
            ds().titoli(xtag='P [uW]', ytag = 'T [k]', titolo='with quality filter')
            for i in range(T_quality_selected.shape[0]):
                ds().dati(x = T['xx0'], y = T_quality_selected.iloc[i], colore=palette[i], scat_plot ='scat', larghezza_riga =15)
            ds().dati(x = T['xx0'], y = average_T_quality_selected['Temp'], colore=colore, scat_plot = 'scat', larghezza_riga =12)
            ds().dati(x = T['xx0'], y = average_T_quality_selected['Temp'], x_error = x_err, y_error= average_T_quality_selected['Err Temp'], colore=colore, scat_plot = 'err')
            ds().dati(x_tot, y_fit2, colore=colore, descrizione='Y = '+str(round(m,2)) + '*X + ' + str(round(q,2))+'\n'+
                      'm = '+str(round(m,2))+' +/- '+ str(round(m_err,2))+'\n q = '+str(round(q,2))+' +/- '+ str(round(q_err,2)))
            plt.fill_between(x_tot, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
            ds().legenda()

        plot2(T, T_quality_selected, average_T_quality_selected, 'blue')
        plot2(T2, T_quality_selected2, average_T_quality_selected2, 'red')
        ds().nuova_fig(41)
        st.pyplot()

        # ██████   █████  ██████      ██████  ██       ██████  ████████
        # ██   ██ ██   ██ ██   ██     ██   ██ ██      ██    ██    ██
        # ██████  ███████ ██████      ██████  ██      ██    ██    ██
        # ██   ██ ██   ██ ██   ██     ██      ██      ██    ██    ██
        # ██████  ██   ██ ██   ██     ██      ███████  ██████     ██


    def bar_plot(self, data_row):
        empty_plot1 = st.empty()
        empty_plot2 = st.empty()
        num_mat = len(data_row['material'].unique().tolist())
        i = 0
        xx = pd.DataFrame({'radius_nm': data_row['radius_nm'].unique()})
        xx = xx['radius_nm'].to_numpy()
        yy = np.zeros(len(xx))

        for material in data_row['material'].unique().tolist():
            data = data_row[data_row['material'] == material]
            data_medie = pd.DataFrame({'radius_nm': data['radius_nm'].unique()})
            raggi = data['radius_nm'].unique().tolist()

            media_quality = []
            media_speed = []
            media_error_q = []
            media_error_s = []
            media_material = []

            for raggio in raggi:
                media_quality.append(data['normalized_signal_quality'][data['radius_nm'] == raggio].mean())
                media_speed.append(data['signal_speed'][data['radius_nm'] == raggio].mean())
                media_error_q.append(data['normalized_signal_quality'][data['radius_nm'] == raggio].std())
                media_error_s.append(data['signal_speed'][data['radius_nm'] == raggio].std())
                media_material.append(material)

            data_medie['normalized_signal_quality'] = media_quality
            data_medie['signal_speed'] = media_speed
            data_medie['normalized_signal_quality_err'] = media_error_q
            data_medie['signal_speed_err'] = media_error_s
            data_medie['material'] = media_material

            data_medie = data_medie.fillna(0)

            st.write(material)
            st.table(data)

            delta_delay = (12)/num_mat
            delay = -3 + delta_delay*i

            ds().nuova_fig(30)
            ds().titoli(titolo='Normalized Signal Intensity', xtag='radius[nm]', ytag='counts')
            ds().dati(x = data_medie['radius_nm'].to_numpy(), y = data_medie['normalized_signal_quality'].to_numpy(), scat_plot = 'bar', delay = delay, width = 3, descrizione=material)
            ds().dati(x = data_medie['radius_nm'].to_numpy()+delay/2, y = data_medie['normalized_signal_quality'].to_numpy(), y_error=data_medie['normalized_signal_quality_err'].to_numpy()/2, scat_plot = 'err', colore='black')
            ds().dati(x = data['radius_nm']+delay/2, y = data['normalized_signal_quality'], scat_plot ='scat', colore="blue", larghezza_riga =12, layer = 2)
            ds().dati(x = xx, y = yy, scat_plot ='bar', width = 3, delay = 0)
            ds().legenda()

            ds().nuova_fig(31)
            ds().titoli(titolo='Slope (C)', xtag='radius[nm]', ytag='T/I [k/uW]')
            ds().dati(x = data_medie['radius_nm'].to_numpy(), y = data_medie['signal_speed'].to_numpy(), scat_plot = 'bar', delay = delay, width = 3, descrizione=material)
            ds().dati(x = data_medie['radius_nm'].to_numpy()+delay/2, y = data_medie['signal_speed'].to_numpy(), y_error=data_medie['signal_speed_err'].to_numpy()/2, scat_plot = 'err', colore='black')
            ds().dati(x = data['radius_nm']+delay/2, y = data['signal_speed'], scat_plot ='scat', colore="blue", larghezza_riga =12, layer = 2)
            ds().dati(x = xx, y = yy, scat_plot ='bar', width = 3, delay = 0)
            ds().legenda()
            i = i+1

        ds().nuova_fig(30)
        empty_plot1.pyplot()

        ds().nuova_fig(31)
        empty_plot2.pyplot()
