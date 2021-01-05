from AJ_draw import disegna as ds
from AJ_function_4_analysis import size_manipulator as sm
from AJ_workflow_preprocess import preprocess as pp

from AJ_PL_analisi import normalization as nr
from AJ_PL_analisi import pulizzia as pl
from AJ_PL_analisi import spectral_anaisi as sa
from AJ_PL_analisi import intensity_map as im
import numpy as np
import pandas as pd
import streamlit as st
import base64

from scipy.stats.distributions import t as tstud
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

palette = ['#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']

class mapping:
    """
    :param file_lamp: file of the lamp Spectrum
    :param file_bkg_single_disks: file of the background for the dark field of the single disk
    :param file_bkg_laser: file of the background for the photoluminescence
    :param file_disk: file of the single disk dark field Spectrum
    :param file_PL: file of the photoluminescence spectrum
    :param num_of_scan: number of scan in the 'file_PL'
    """
    def __init__(self, raw_data, num_of_scan, bkg_position = 'min', remove_cosmic_ray = 'no', soglia_derivata = 0, nome_file = 'dati', x_bkg = -1, y_bkg = -1, material = 'au', log_scale = 0):
        # st.text(nome_file)
        self.raw_data = raw_data
        self.num_of_scan = num_of_scan
        self.remove_cosmic_ray = remove_cosmic_ray
        self.nome_file = nome_file
        self.soglia_derivata = soglia_derivata
        self.bkg_position = bkg_position
        self.x_bkg = x_bkg
        self.y_bkg = y_bkg
        self.material = material
        self.log_scale = log_scale


    def good_to_go(self, AS_min = 590, AS_max = 616, S_min = 645, S_max = 699, wave_inf = 591, wave_sup = 616, num_bin = 10, numb_of_spect = 10, laser_power = 1.75, raggio = 0.07, lato_cella = 0.05,
                   riga_y_start = 0, riga_y_stop = 1, pixel_0 = [3,0], temp_max = 550, T_RT = 295, laser_type = 633, salva = 'no', cut_min = 0, cut_max = 0, selected_scan = 0, punti_cross_section_geometrica = 10):

        def download_file(data, filename):
            testo = 'Download '+filename+'.csv'
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">'+testo+'</a>'
            st.markdown(href, unsafe_allow_html=True)


        """
        calculated the heat map
        wave_inf e wave_sup rappresentano l'intervallo in cui viene calcolato lo stokes per la heatmap
        """

        # ██       ██████   █████  ██████
        # ██      ██    ██ ██   ██ ██   ██
        # ██      ██    ██ ███████ ██   ██
        # ██      ██    ██ ██   ██ ██   ██
        # ███████  ██████  ██   ██ ██████

        if self.remove_cosmic_ray == 'yes':
            if salva == 'yes':
                st.text('cosmic ray cleaning')
            raw_PL = pp(self.raw_data, num_of_scan=self.num_of_scan).remove_cosmic_ray(self.soglia_derivata)
        else:
            raw_PL = self.raw_data

        # ██████  ███████ ███    ███  ██████  ██    ██ ███████     ██████  ██   ██  ██████
        # ██   ██ ██      ████  ████ ██    ██ ██    ██ ██          ██   ██ ██  ██  ██
        # ██████  █████   ██ ████ ██ ██    ██ ██    ██ █████       ██████  █████   ██   ███
        # ██   ██ ██      ██  ██  ██ ██    ██  ██  ██  ██          ██   ██ ██  ██  ██    ██
        # ██   ██ ███████ ██      ██  ██████    ████   ███████     ██████  ██   ██  ██████

        if salva == 'yes':
            st.text('remove background')

        PL_x, PL_y = nr(file_data=raw_PL, map_conf ='yes').bkg_map(laser_type = laser_type, bkg_position = self.bkg_position, x_bkg = self.x_bkg, y_bkg = self.y_bkg)

        # ██ ███    ██ ████████ ███████ ███    ██ ███████ ██ ████████ ██    ██     ███    ███  █████  ██████
        # ██ ████   ██    ██    ██      ████   ██ ██      ██    ██     ██  ██      ████  ████ ██   ██ ██   ██
        # ██ ██ ██  ██    ██    █████   ██ ██  ██ ███████ ██    ██      ████       ██ ████ ██ ███████ ██████
        # ██ ██  ██ ██    ██    ██      ██  ██ ██      ██ ██    ██       ██        ██  ██  ██ ██   ██ ██
        # ██ ██   ████    ██    ███████ ██   ████ ███████ ██    ██       ██        ██      ██ ██   ██ ██

        if salva == 'yes':
            st.text('creating the intensity map')

        # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        #sostituisce un certo numero di righe sulla intensity map con l'intensita del pixel_0
        #serve per eliminare il rumore di fondo quando e' troppo alto in picocle porzioni del grafico
        for i in range(0, len(PL_y[0,:,0])):# X
            for j in range(riga_y_start, riga_y_stop):# Y
                PL_y[:,j,i] = PL_y[:, pixel_0[0], pixel_0[1]]
        # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

        intensity_sp = np.zeros((len(PL_y[0,:,0]), len(PL_y[0,0,:])))

        for k in range(len(PL_y[0,:,0])):
            for i in range(len(PL_x)):
                if wave_inf>PL_x[i]:
                    i591 = i
            for i in range(len(PL_x)):
                if wave_sup>PL_x[i]:
                    i616 = i
            for j in range(self.num_of_scan):
                sum_temp = 0
                for i in range(i591,i616):
                    sum_temp =  PL_y[i,k,j] + sum_temp
                sum_temp = sum_temp/(i616-i591)
                intensity_sp[k,j] = sum_temp

        x = [i*lato_cella for i in range(len(intensity_sp[:,0]))]
        y = [i*lato_cella for i in range(len(intensity_sp[0,:]))]
        xy = np.meshgrid(x,y)

        #####################################################################
        if salva == 'yes':
            ds().nuova_fig(indice_fig=4)
            ds().titoli(titolo='', xtag='um', ytag='um')
            ds().dati(x, y, scat_plot ='cmap', z =intensity_sp)

            ds().nuova_fig(indice_fig=2, indice_subplot =321, width =15, height = 10)
            ds().titoli(titolo=self.nome_file, xtag='um', ytag='um')
            ds().dati(x, y, scat_plot ='cmap', z =intensity_sp)
            #####################################################################

        # ███████  ██████  ██████  ████████     ██ ███    ██ ████████ ███████ ███    ██ ███████ ██ ████████ ██    ██
        # ██      ██    ██ ██   ██    ██        ██ ████   ██    ██    ██      ████   ██ ██      ██    ██     ██  ██
        # ███████ ██    ██ ██████     ██        ██ ██ ██  ██    ██    █████   ██ ██  ██ ███████ ██    ██      ████
        #      ██ ██    ██ ██   ██    ██        ██ ██  ██ ██    ██    ██      ██  ██ ██      ██ ██    ██       ██
        # ███████  ██████  ██   ██    ██        ██ ██   ████    ██    ███████ ██   ████ ███████ ██    ██       ██

        if salva == 'yes':
            st.text('sorting the intensity')

        sorted_intensity0 = np.zeros(len(PL_y[0,:,0])*len(PL_y[0,0,:]))

        k = 0
        for i in range(len(intensity_sp[:,0])):
            for j in range(len(intensity_sp[0,:])):
                sorted_intensity0[k] = intensity_sp[i,j]
                k = k + 1

        sorted_intensity0 = sorted(sorted_intensity0, reverse = True)

        #####################################################################################
        intens_coord = np.where(intensity_sp == sorted_intensity0[0])
        if len(intens_coord) == 1:
            intens_coord_y = intens_coord[0]
            intens_coord_x = intens_coord[1]
        else:
            intens_coord_y = intens_coord[0][0]
            intens_coord_x = intens_coord[1][0]

        # xc = x[intens_coord_x]
        # yc = y[intens_coord_y]
        # intensity_sp2 = im(salva).fit_guass(x=x, y=y, xy_mesh= xy, amp=sorted_intensity0[0], intensity_sp=intensity_sp, xc=xc, yc=yc, rad = raggio, laser_power=laser_power, punti_cross_section_geometrica = punti_cross_section_geometrica)

        intensity_sp2 = im(salva).eval_power(xy_mesh= xy, intensity_sp=intensity_sp, laser_power=laser_power)
        #####################################################################################

        sorted_intensity = np.zeros(len(PL_y[0,:,0])*len(PL_y[0,0,:]))

        k = 0
        for i in range(len(intensity_sp2[:,0])):
            for j in range(len(intensity_sp2[0,:])):
                sorted_intensity[k] = intensity_sp2[i,j]
                k = k + 1

        sorted_intensity = sorted(sorted_intensity, reverse = True)
        # tot_intensity = sum(sorted_intensity)

        # ██████  ██ ███    ██ ███    ██ ██ ███    ██  ██████
        # ██   ██ ██ ████   ██ ████   ██ ██ ████   ██ ██
        # ██████  ██ ██ ██  ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
        # ██   ██ ██ ██  ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
        # ██████  ██ ██   ████ ██   ████ ██ ██   ████  ██████

        if salva == 'yes':
            st.text('binning')

        num_divisions = int(len(sorted_intensity)/num_bin)
        PL_bin = np.zeros((len(PL_y[:,0,0]), num_divisions))
        average_intensity = np.zeros(num_divisions)
        average_intensity_vet = np.zeros(num_bin)
        average_intensity_err = np.zeros((2, num_divisions))

        coordX_intesita = np.zeros((num_divisions, num_bin))
        coordY_intesita = np.zeros((num_divisions, num_bin))

        k = 0
        for i in range(num_divisions):
            for j in range(num_bin):
                intens_coord = np.where(intensity_sp2 == sorted_intensity[k])
                if len(intens_coord) == 1:
                    intens_coord_y = intens_coord[0]
                    intens_coord_x = intens_coord[1]
                else:
                    intens_coord_y = intens_coord[0][0]
                    intens_coord_x = intens_coord[1][0]

                coordX_intesita[i,j] = x[intens_coord_x] + 0.025#è normale che siano invertiti x e y
                coordY_intesita[i,j] = y[intens_coord_y] + 0.025

                for l in range(len(PL_y[:,0,0])):
                    PL_bin[l,i] = PL_y[l, intens_coord_y, intens_coord_x] + PL_bin[l,i]

                average_intensity_vet[j] = intensity_sp2[intens_coord_y, intens_coord_x]
                average_intensity[i] = intensity_sp2[intens_coord_y, intens_coord_x] + average_intensity[i]
                k = k + 1

            average_intensity[i] = average_intensity[i]/num_bin

            average_intensity_err[0,i] = average_intensity[i] - min(average_intensity_vet)
            average_intensity_err[1,i] = max(average_intensity_vet) - average_intensity[i]

        if salva == 'yes':
            ds().nuova_fig(indice_fig=4)
            for i in range(numb_of_spect):
                ds().dati(coordX_intesita[i,:], coordY_intesita[i,:], scat_plot = 'scat', colore=palette[i], larghezza_riga =30, layer =2)
            st.pyplot()

            ds().nuova_fig(indice_fig=2, indice_subplot =321)
            for i in range(numb_of_spect):
                ds().dati(coordX_intesita[i,:], coordY_intesita[i,:], scat_plot = 'scat', colore=palette[i], larghezza_riga =30, layer =2)


        # ██████  ███████ ███████ ██   ██  █████  ██████  ███████
        # ██   ██ ██      ██      ██   ██ ██   ██ ██   ██ ██
        # ██████  █████   ███████ ███████ ███████ ██████  █████
        # ██   ██ ██           ██ ██   ██ ██   ██ ██      ██
        # ██   ██ ███████ ███████ ██   ██ ██   ██ ██      ███████

        if salva == 'yes':
            st.text('select only the chosen number of spectrum')

        col_num = [i for i in range(numb_of_spect)]
        PL_bin = sm(PL_bin).rescape(col_num)

        #the code here add to all the selected spectrum the smallest value in the anistokes, so no spectrum goes below 0
        #still experimenting with this
        # for i in range(len(PL_x)):
        #     if AS_min>PL_x[i]:
        #         iAS_min = i
        #     if AS_max>PL_x[i]:
        #         iAS_max = i
        # PL_bin = PL_bin + np.abs(PL_bin[iAS_min:iAS_max,:].min())

        average_intensity_new = np.zeros(numb_of_spect)
        average_intensity_err_new = [[i]*numb_of_spect for i in range(2)]

        average_intensity_new = [average_intensity[i] for i in range(numb_of_spect)]
        for i in range(2):
            for j in range(numb_of_spect):
                average_intensity_err_new[i][j] = average_intensity_err[i,j]

        # ███████ ███    ███  ██████   ██████  ████████ ██   ██
        # ██      ████  ████ ██    ██ ██    ██    ██    ██   ██
        # ███████ ██ ████ ██ ██    ██ ██    ██    ██    ███████
        #      ██ ██  ██  ██ ██    ██ ██    ██    ██    ██   ██
        # ███████ ██      ██  ██████   ██████     ██    ██   ██

        if salva == 'yes':
            st.text('smoothing')

        ncycl = 4
        y_smooth = pl(PL_x, PL_bin, len(PL_bin[0])).smoothing_fft(ncycl)
        y_smooth = pl(PL_x, y_smooth, len(PL_bin[0])).smoothing_fft(ncycl)
        y_smooth = pl(PL_x, y_smooth, len(PL_bin[0])).smoothing_fft(ncycl)
        y_smooth = pl(PL_x, y_smooth, len(PL_bin[0])).smoothing_fft(ncycl)

        #####################################################################
        i_pl = 0
        for i in range(len(PL_x)):
            if PL_x[i] < S_min+1:
                i_pl = i

        if salva == 'yes':
            ds().nuova_fig(indice_fig=5)
            ds().titoli(titolo='', xtag="nm", ytag='count')
            for i in range(len(PL_bin[0,:])):
                ds().dati(PL_x, y_smooth[:,i], colore=palette[i], descrizione= str(i))
            ds().range_plot(bottomX=AS_min, topX=S_max, bottomY=-10,  topY=100+max(y_smooth[i_pl:,0]))
            # ds().range_plot(bottomX=AS_min, topX=AS_max, bottomY=-10,  topY=1000)

            save_plot = pd.DataFrame()
            save_plot['x'] = PL_x
            for i in range(len(PL_bin[0,:])):
                save_plot[i] = y_smooth[:,i]
            download_file(save_plot, self.nome_file + '_PL')
            st.pyplot()

            ds().nuova_fig(indice_fig=2, indice_subplot =323)
            ds().titoli(titolo='', xtag="nm", ytag='count')
            for i in range(len(PL_bin[0,:])):
                ds().dati(PL_x, y_smooth[:,i], colore=palette[i], descrizione= str(i))
            ds().range_plot(bottomX=AS_min, topX=S_max, bottomY=-10,  topY=100+max(y_smooth[i_pl:,0]))

        signal_quality = max(y_smooth[i_pl:,0])
        #####################################################################

        # ██████   █████  ████████ ██  ██████
        # ██   ██ ██   ██    ██    ██ ██    ██
        # ██████  ███████    ██    ██ ██    ██
        # ██   ██ ██   ██    ██    ██ ██    ██
        # ██   ██ ██   ██    ██    ██  ██████

        # y_smooth = np.zeros((len(self.data_y), self.number_of_scan))

        if salva == 'yes':
            st.text('creating the ratio')
        PL_ratio_raman = sa(PL_x, y_smooth, salva, len(y_smooth[0]), temp_max = temp_max).power_ratio_plain(AS_min = AS_min, AS_max = AS_max, S_min =S_min, S_max =S_max, cut_min = cut_min, cut_max = cut_max, selected_scan =selected_scan)
        if salva == 'yes':
            st.text('fit with the raman function')
        T_raman, R2_raman, _, ET_raman, _ = sa(PL_ratio_raman[0], PL_ratio_raman[1], salva, len(y_smooth[0]), temp_max = temp_max).temperature_Raman(S_min =S_min, T_RT = T_RT, laser_type = laser_type)

        if salva == 'yes':
            save_plot = pd.DataFrame()
            save_plot['x'] = PL_ratio_raman[0]
            for i in range(PL_ratio_raman[1].shape[1]):
                save_plot[i] = PL_ratio_raman[1][:,i]
            download_file(save_plot, self.nome_file + '_ratio')

        T_dense, ET_dense = sa(PL_ratio_raman[0], PL_ratio_raman[1], salva, len(y_smooth[0]), temp_max = temp_max).ratio_vs_power(average_intensity_new, S_min = S_min, S_max = S_max, AS_min = AS_min, AS_max = AS_max,
                                                                                                                                  T_RT = T_RT, laser_type = laser_type, selected_scan = selected_scan)
        if salva == 'yes':
            st.text('Logaritmic scale')
        controllo_negativi = 0
        for i in range(PL_ratio_raman[1].shape[0]*PL_ratio_raman[1].shape[1]):
            controllo_temp = PL_ratio_raman[1].flat[i]
            if controllo_temp <= 0:
                controllo_negativi = 1
        if controllo_negativi == 0:
            T_log, R2_log, _, ET_log, _ = sa(PL_ratio_raman[0], PL_ratio_raman[1], salva, len(y_smooth[0]), temp_max = temp_max).log_ratio(S_min = S_max, S_max = S_max, AS_min = AS_min, AS_max = AS_max, T_RT = T_RT, laser_type = laser_type)
        else:
            if self.log_scale == 1:
                self.log_scale = 2
                st.error('impossible calculate the log, a negative number is present, standard scale is used instead')
        #### y_nan = sm(PL_ratio_y).nansubstitute(PL_ratio_x)

        if self.log_scale == 1:
            T_raman = T_log
            R2_raman = R2_log
            ET_raman = ET_log
        elif self.log_scale == 2:
            T_raman = T_dense.tolist()
            R2_raman = [1 for i in range(len(T_dense))]
            ET_raman = ET_dense.tolist()

        # ███████ ██ ████████
        # ██      ██    ██
        # █████   ██    ██
        # ██      ██    ██
        # ██      ██    ██

        if salva == 'yes':
            st.text('fit the temperature with the power dependency')
        def retta(x, p0, p1):
            return p0*x + p1

        x1 = np.array(average_intensity_new, dtype="float")
        y2 = np.array(T_raman, dtype="float")
        par1, par2 = curve_fit(retta, x1, y2)
        y_fit2 = retta(x1, par1[0], par1[1])

        m = par1[0]
        q = par1[1]
        # residual = y2 - y_fit2
        # ss_res = np.sum(residual**2)
        ss_tot = np.sum((y2 - np.mean(y2))**2)
        if ss_tot == 0:
            ss_tot = 1
            # ss_res = 1
        # r2 = 1- (ss_res/ss_tot)
        # p = len(par1)
        # n = len(x1)
        alpha = 0.05 #95% confidence interval
        dof = max(0, len(x1) - len(par1)) #degree of freedom
        tval = tstud.ppf(1.0 - alpha/2., dof) #t-student value for the dof and confidence level
        sigma = np.diag(par2)**0.5
        m_err = sigma[0]*tval
        q_err = sigma[1]*tval

        y_fit2_up = retta(x1, m+(m_err/2), q+(q_err/2))
        y_fit2_down = retta(x1, m-(m_err/2), q-(q_err/2))

        #####################################################################
        if salva == 'yes':
            ds().nuova_fig(indice_fig=6)
            # ds().titoli(xtag='I [mW/um^2]', ytag = 'T [k]', titolo='')
            ds().titoli(xtag='P [uW]', ytag='T [k]', titolo='')
            ds().dati(average_intensity_new, T_raman, x_error = average_intensity_err_new, y_error = ET_raman, scat_plot = 'err')
            ds().dati(x1, y_fit2, colore='black', descrizione=str(round(par1[0],2)) + '*X + ' + str(round(par1[1],2))+'\n'+
                      str(round(m,2))+' +/- '+ str(round(m_err,2))+'\n'+str(round(q,2))+' +/- '+ str(round(q_err,2)))
            plt.fill_between(x1, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
            ds().legenda()
            st.pyplot()

            ds().nuova_fig(indice_fig=2, indice_subplot =325)
            # ds().titoli(xtag='I [mW/um^2]', ytag = 'T [k]', titolo='')
            ds().titoli(xtag='P [uW]', ytag='T [k]', titolo='')
            ds().dati(average_intensity_new, T_raman, x_error = average_intensity_err_new, y_error = ET_raman, scat_plot = 'err')
            ds().dati(x1, y_fit2, colore='black', descrizione=str(round(par1[0],2)) + '*X + ' + str(round(par1[1],2))+'\n'+
                      str(round(m,2))+' +/- '+ str(round(m_err,2))+'\n'+str(round(q,2))+' +/- '+ str(round(q_err,2)))
            plt.fill_between(x1, y_fit2_down, y_fit2_up, color = 'black', alpha = 0.15)
            ds().legenda()

        # if salva != 'yes':
        #     ds().porta_a_finestra(chiudi = 1)
        #####################################################################

        # ███████  █████  ██    ██ ███████
        # ██      ██   ██ ██    ██ ██
        # ███████ ███████ ██    ██ █████
        #      ██ ██   ██  ██  ██  ██
        # ███████ ██   ██   ████   ███████

        matrice_salve2 = pd.DataFrame()
        matrice_salve2['average_intensity_new'] = average_intensity_new
        matrice_salve2['Temperature'] = T_raman
        matrice_salve2['average_intensity_err_low'] = average_intensity_err_new[0]
        matrice_salve2['average_intensity_err_up'] = average_intensity_err_new[1]
        matrice_salve2['Temperature_err'] = ET_raman
        matrice_salve2['Temperature_r2'] = R2_raman

        matrice_salve3 = pd.DataFrame()
        matrice_salve3['signal_quality'] = [signal_quality]
        matrice_salve3['signal_speed'] = [round(par1[0],2)] #coefficiente angolare
        matrice_salve3['radius'] = [raggio]
        matrice_salve3['laser_type'] = [laser_type]
        matrice_salve3['laser_power'] = [laser_power]
        matrice_salve3['material'] = [self.material]

        if salva == 'yes':
            st.pyplot()
        return par1[1], matrice_salve2, matrice_salve3
