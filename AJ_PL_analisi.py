import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import t as tstud
from AJ_function_4_analysis import size_manipulator as sm
from AJ_draw import disegna as ds
import pandas as pd
import streamlit as st
palette = ['#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']


# ██████   █████  ████████  █████          ███████ ████████  ██████  ██████   █████   ██████  ███████
# ██   ██ ██   ██    ██    ██   ██         ██         ██    ██    ██ ██   ██ ██   ██ ██       ██
# ██   ██ ███████    ██    ███████         ███████    ██    ██    ██ ██████  ███████ ██   ███ █████
# ██   ██ ██   ██    ██    ██   ██              ██    ██    ██    ██ ██   ██ ██   ██ ██    ██ ██
# ██████  ██   ██    ██    ██   ██ ███████ ███████    ██     ██████  ██   ██ ██   ██  ██████  ███████


class data_storage:

    def __init__(self, file):
        self.file = file

    def load_map(self, number_of_scan=1):
        file  = self.file
        file = pd.DataFrame(file)
        file = file.drop([0, 1], axis=0)

        file = file[0].str.split(expand=True)
        file = file.apply(pd.to_numeric, downcast='float').to_numpy()
        data_x = file[:,0]
        data_y_raw = file[:,1:]

        data_y = np.zeros((len(data_x), number_of_scan, number_of_scan))
        kk = 0
        for i in range(number_of_scan):
            for j in range(number_of_scan):
                data_y[:, i, j] = data_y_raw[:,kk]
                kk = kk + 1
        return data_x, data_y

    def load_map_multiple_files(self, number_of_scan=1):
        file  = self.file
        file = file.drop([0], axis=0)
        file.rename(columns={file.columns.tolist()[0]:0}, inplace=True)

        file = file[0].str.split(expand=True)
        file = file.apply(pd.to_numeric, downcast='float').to_numpy()
        data_x = file[:,0]
        data_y_raw = file[:,1:]

        data_y = np.zeros((len(data_x), number_of_scan, number_of_scan))
        kk = 0
        for i in range(number_of_scan):
            for j in range(number_of_scan):
                data_y[:, i, j] = data_y_raw[:,kk]
                kk = kk + 1
        return data_x, data_y

        # ███    ██  ██████  ██████  ███    ███  █████  ██      ██ ███████  █████  ████████ ██  ██████  ███    ██
        # ████   ██ ██    ██ ██   ██ ████  ████ ██   ██ ██      ██    ███  ██   ██    ██    ██ ██    ██ ████   ██
        # ██ ██  ██ ██    ██ ██████  ██ ████ ██ ███████ ██      ██   ███   ███████    ██    ██ ██    ██ ██ ██  ██
        # ██  ██ ██ ██    ██ ██   ██ ██  ██  ██ ██   ██ ██      ██  ███    ██   ██    ██    ██ ██    ██ ██  ██ ██
        # ██   ████  ██████  ██   ██ ██      ██ ██   ██ ███████ ██ ███████ ██   ██    ██    ██  ██████  ██   ████


class normalization:
    """
    :param file_data_x: file name with the data where the first column is the X coordinate and the other columns are the Y
    :param file_lamp_x: file name with the data of the lamp spectrum where the first colunm is the X coordinate and the other columns are the Y
    :param file_lamp_y: if the lamp data are splitted in 2 files, you place here the file name with the y coordinates
    :param file_bkg_x: file name with the data of the background spectrum where the first colunm is the X coordinate and the other columns are the Y
    :param file_bkg_y: if the background data are splitted in 2 files, you place here the file name with the y coordinates
    :param separator: useless parameter
    :param reduced: 'yes' or 'no', if 'yes' you can choose a reduced number of scan otherwies all are taken
    :param number_of_scan: number of scan in the file of the data
    """
    def __init__(self, file_data, map_conf = 'no'):

        self.number_of_scan = len(file_data[1][0,:])

        if map_conf == 'yes':
            self.data_x = np.zeros(len(file_data[0]))
            self.data_y = np.zeros((len(file_data[0]), len(file_data[1][0,:,0]), len(file_data[1][0,0,:])))

            self.data_x[:] = file_data[0][:]
            self.data_y[:,:,:] = file_data[1][:,:,:]
        else:
            self.data_x = np.zeros(len(file_data[0]))
            self.data_y = np.zeros((len(file_data[0]), len(file_data[1][0,:])))

            self.data_x[:] = file_data[0][:]
            self.data_y[:,:] = file_data[1][:,:]

    def bkg_map(self, laser_type = 633, bkg_position = 'min', x_bkg = -1, y_bkg = -1):

        # intensity_sp = np.zeros((len(self.data_y[0,:,0]),len(self.data_y[0,0,:])))

        i650 = laser_type + 20
        i680 = i650 + 30

        for i in range(len(self.data_x)):
            if i650>self.data_x[i]:
                i591 = i

        for i in range(len(self.data_x)):
            if i680>self.data_x[i]:
                i616 = i

        sum_temp_matr = np.zeros((len(self.data_y[0,:,0]),len(self.data_y[0,0,:])))

        for i in range(len(self.data_y[0,:,0])):
            for j in range(len(self.data_y[0,0,:])):
                sum_temp = 0
                for k in range(i591, i616):
                    sum_temp = self.data_y[k,i,j] + sum_temp
                sum_temp_matr[i,j] = sum_temp

        x_min, y_min = np.where(sum_temp_matr == sum_temp_matr.min())

        if bkg_position == 'min':
            bkg_temp = self.data_y[:,x_min[0],y_min[0]].copy()
        else:
            bkg_temp = self.data_y[:,x_bkg,y_bkg].copy()

        for i in range(len(self.data_y[0,:,0])):
            for j in range(len(self.data_y[0,0,:])):
                if bkg_position == 'min':
                    self.data_y[:,i,j] = self.data_y[:,i,j] - bkg_temp
                else:
                    self.data_y[:,i,j] = self.data_y[:,i,j] - bkg_temp

        return self.data_x, self.data_y


        # ██████  ██    ██ ██      ██ ███████ ███████ ██  █████
        # ██   ██ ██    ██ ██      ██    ███     ███  ██ ██   ██
        # ██████  ██    ██ ██      ██   ███     ███   ██ ███████
        # ██      ██    ██ ██      ██  ███     ███    ██ ██   ██
        # ██       ██████  ███████ ██ ███████ ███████ ██ ██   ██

class pulizzia:
    """
    :param data_x: array with the X coordinates
    :param data_y: matrix with the Y coordinates of all the sans
    :param number_of_scan: number of scans present in the data_y
    """
    def __init__(self, data_x, data_y, number_of_scan=1):
        self.data_x = data_x
        self.data_y = data_y
        self.number_of_scan = number_of_scan

    def smoothing_fft(self, punti_per_box):
        """
        Smoothing function using the FFT

        :param punti_per_box: number of point you want to average

        :return: Y[len(data_y), number_of_scan]
        """
        box = np.ones(punti_per_box)/punti_per_box
        y_smooth = np.zeros((len(self.data_y), self.number_of_scan))
        for i in range(0, self.number_of_scan):
            y_smooth[:, i] = np.convolve(self.data_y[:, i], box, mode='same')
        return y_smooth


    def salva_dati(self, nomefile='dati_PL.txt'):
        """
        function to save the data into a file

        :param nomefile: name of the output file
        """

        file = open(nomefile, 'w')
        for i in range(len(self.data_x)):
            file.write(str(self.data_x[i])+' ')
            for j in range(self.number_of_scan):
                file.write(str(self.data_y[i, j])+' ')
            file.write('\n')
        file.close()


        # ███████ ██████  ███████  ██████ ████████ ██████   █████  ██               █████  ███    ██  █████  ██ ███████ ██
        # ██      ██   ██ ██      ██         ██    ██   ██ ██   ██ ██              ██   ██ ████   ██ ██   ██ ██ ██      ██
        # ███████ ██████  █████   ██         ██    ██████  ███████ ██              ███████ ██ ██  ██ ███████ ██ ███████ ██
        #      ██ ██      ██      ██         ██    ██   ██ ██   ██ ██              ██   ██ ██  ██ ██ ██   ██ ██      ██ ██
        # ███████ ██      ███████  ██████    ██    ██   ██ ██   ██ ███████ ███████ ██   ██ ██   ████ ██   ██ ██ ███████ ██



class spectral_anaisi:
    """
    :param data_x: array with the X coordinates
    :param data_y: matrix with the Y coordinates of all the scans
    :param number_of_scan: number of scans present in the data_y
    :param fix_numeratore: 'yes' or 'no', if 'yes' the ratio is calculated keeping fixed the numerator, with 'no' the ratio is calculated keeping fixed the denominator
    """
    def __init__(self, data_x, data_y, salva, number_of_scan=1, temp_max = 550):
        self.data_x = data_x
        self.data_y = data_y
        self.salva = salva
        self.number_of_scan = number_of_scan
        self.temp_max = temp_max
        self.divP = 5 #serve per delineare il range in cui viene limitato il fit sulla potenza

    def cut_data_optimal_range(self, AS_min, AS_max, S_min, S_max):
        AS_min_lab = 0
        AS_max_lab = 0
        S_min_lab = 0
        S_max_lab = 0

        for i in range(len(self.data_x)):
            if self.data_x[i] <= AS_min:
                AS_min_lab = i
            if self.data_x[i] <= AS_max:
                AS_max_lab = i
            if self.data_x[i] <= S_min:
                S_min_lab = i
            if self.data_x[i] <= S_max:
                S_max_lab = i

        data_cut_x = np.zeros(AS_max_lab - AS_min_lab + 1 + S_max_lab - S_min_lab + 1)
        data_cut_y = np.zeros((AS_max_lab - AS_min_lab + 1 + S_max_lab - S_min_lab + 1, round(self.number_of_scan)))

        step = 0
        for i in range(AS_min_lab, AS_max_lab + 1):
            data_cut_x[step] = self.data_x[i]
            data_cut_y[step,:] = self.data_y[i,:]
            step = step + 1

        step = 0
        for i in range(S_min_lab, S_max_lab + 1):
            data_cut_x[step + AS_max_lab - AS_min_lab + 1] = self.data_x[i]
            data_cut_y[step + AS_max_lab - AS_min_lab + 1,:] = self.data_y[i,:]
            step = step + 1
        return data_cut_x, data_cut_y

    def power_ratio_plain(self, AS_min = 588, AS_max = 616, S_min = 645, S_max = 699, cut_min = 0, cut_max = 0, selected_scan = 0):
        """
        This function divides the spectrum of a matrix to get the ratio values, in order to find the temperature.

        :param AS_min: value of the Anti-Stoke lower limit
        :param AS_max: value of the Anti-Stoke upper limit
        :param S_min: value of the Stoke lower limit
        :param S_max: value of the Stoke upper limit
        :param direzione: 'top' or 'bottom', use 'top' if the first spectrum is the most intense, use 'bottom' if the last spectrum is the most intense

        The final length is always smaller then the original vector, because the area with the laser is removed

        :return: X[:], Y[len(X), num_of_scan]
        """

        data_cut_x, data_cut_y = self.cut_data_optimal_range(AS_min, AS_max, S_min, S_max)
        data_ratio = np.zeros((data_cut_y.shape[0], data_cut_y.shape[1]))
        for i in range(self.number_of_scan):
            data_ratio[:,i] = data_cut_y[:,i]/data_cut_y[:,self.number_of_scan-1-selected_scan]

        if cut_min == 0:
            return data_cut_x, data_ratio

        else:
            for i in range(len(data_cut_x)):
                if data_cut_x[i] <= cut_min:#753
                    AS_cut_min = i
                if data_cut_x[i] <= cut_max:#755.5
                    AS_cut_max = i

            data_cut_final_x = []
            for i in range(len(data_cut_x)):
                if i < AS_cut_min or i > AS_cut_max:
                    data_cut_final_x.append(data_cut_x[i])

            data_cut_final_y = np.zeros((len(data_cut_final_x), (self.number_of_scan)))
            j = 0
            for i in range(len(data_cut_x)):
                if i < AS_cut_min or i > AS_cut_max:
                    data_cut_final_y[j,:] = data_ratio[i,:]
                    j = j + 1
        return data_cut_final_x, data_cut_final_y


    def temperature_Raman(self, S_min = 645, T_RT = 295, laser_type = 633):
        def funzione_raman(x , p1, p0):
            h_bar = 4.1356*1e-15
            T_0 = T_RT
            kb = 8.617*1e-5
            c = 299792458
            l_laser = laser_type
            w_laser = c/(l_laser*1e-9)
            # w_laser = 0

            numeratore   = np.exp((h_bar*(c/(x*1e-9) - w_laser))/(kb*T_0)) - 1
            denominatore = np.exp((h_bar*(c/(x*1e-9) - w_laser))/(kb*p1)) - 1
            return p0*numeratore/denominatore

        data_cut_x, data_cut_y = self.data_x, self.data_y
        data_cut_x = np.asarray(data_cut_x)

        T1 = np.zeros((self.number_of_scan))
        P1 = np.zeros((self.number_of_scan))
        r2 = np.zeros((self.number_of_scan))
        ET1 =np.zeros((self.number_of_scan))
        EP1 =np.zeros((self.number_of_scan))

        resolution = data_cut_x[1] - data_cut_x[0]
        bandwidth = data_cut_x[-1] - data_cut_x[0]
        num_point = int(round(bandwidth/resolution))
        x_fit = np.zeros(num_point)
        y_fit = np.zeros((num_point, self.number_of_scan))
        for i in range(num_point):
            x_fit[i] = data_cut_x[0]+resolution*i

        for k in range(len(data_cut_x)):
            if data_cut_x[k]<=S_min+1:
                i_x_min_S = k#indice inferiore di dove parte lo Stokes per poi calcolare la potenza media da mettere come condizione nel fit

        data_cut_y = sm(data_cut_y).nansubstitute(data_cut_x)

        for imatr in range(self.number_of_scan):

            potenza_media = 0
            for k in range(i_x_min_S,len(data_cut_x)):
                potenza_media = data_cut_y[k, imatr] + potenza_media
            potenza_media = potenza_media/(len(data_cut_x)-i_x_min_S)

            popt, pcov = curve_fit(funzione_raman, data_cut_x, data_cut_y[:,imatr], bounds=([295, potenza_media-(potenza_media/self.divP)],[self.temp_max, potenza_media+(potenza_media/self.divP)]))
            # popt, pcov = curve_fit(funzione_raman, data_cut_x, data_cut_y[:,imatr], bounds=([295, -np.inf],[self.temp_max, np.inf]))
            y_fit[:,imatr] = funzione_raman(x_fit, popt[0], popt[1])

            T1[imatr] = popt[0]
            P1[imatr] = popt[1]

            residual = data_cut_y[:,imatr] - funzione_raman(data_cut_x,*popt)
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((data_cut_y[:,imatr] - np.mean(data_cut_y[:,imatr]))**2)
            if ss_tot == 0:
                ss_tot = 1
                ss_res = 1
            r2[imatr] = 1- (ss_res/ss_tot)



            ############################################################
            # p = len(popt)
            # n = len(data_cut_x)
            alpha = 0.05 #95% confidence interval
            dof = max(0, len(data_cut_x) - len(popt)) #degree of freedom
            tval = tstud.ppf(1.0 - alpha/2., dof) #t-student value for the dof and confidence level
            sigma = np.diag(pcov)**0.5
            ET1[imatr] = sigma[0]*tval
            EP1[imatr] = sigma[1]*tval
            ########################################################

        if self.salva == 'yes':
            ds().nuova_fig(indice_fig=7)
            ds().titoli(xtag='nm', ytag='ratio', titolo='')
            for i in range(self.number_of_scan):
                ds().dati(data_cut_x, data_cut_y[:,i], colore= palette[i])
                ds().dati(x_fit, y_fit[:,i], colore=palette[i])
            # st.pyplot()

            ds().nuova_fig(indice_fig=2, indice_subplot = 324)
            ds().titoli(xtag='nm', ytag='ratio', titolo='')
            for i in range(self.number_of_scan):
                ds().dati(data_cut_x, data_cut_y[:,i], colore= palette[i])
                ds().dati(x_fit, y_fit[:,i], colore=palette[i])

        return T1, r2, P1, ET1, EP1

    def log_ratio(self, AS_min = 588, AS_max = 616, S_min = 645, S_max = 699, T_RT = 295, laser_type = 633):
        def funzione_raman2(x , p1, p0):
            kb = 8.617*1e-5
            h_bar = 4.1356*1e-15
            c = 299792458
            return (((h_bar*c)/(kb*1e-9))*((1/T_RT)-(1/p1)))/x + (((h_bar*c)/(kb*1e-9*laser_type))*((1/p1)-(1/T_RT))) + p0

        data_cut_x, data_cut_y = self.cut_data_optimal_range(AS_min, AS_max, S_min, S_max)
        data_cut_x = np.delete(data_cut_x, -1)
        data_cut_y = np.delete(data_cut_y, -1, 0)
        data_cut_y = np.log(data_cut_y)

        T1 = np.zeros((self.number_of_scan))
        P1 = np.zeros((self.number_of_scan))
        r2 = np.zeros((self.number_of_scan))
        ET1 =np.zeros((self.number_of_scan))
        EP1 =np.zeros((self.number_of_scan))

        resolution = data_cut_x[1] - data_cut_x[0]
        bandwidth = data_cut_x[-1] - data_cut_x[0]
        num_point = int(round(bandwidth/resolution))
        x_fit = np.zeros(num_point)
        y_fit = np.zeros((num_point, self.number_of_scan))
        for i in range(num_point):
            x_fit[i] = data_cut_x[0]+resolution*i
        data_cut_y = sm(data_cut_y).nansubstitute(data_cut_x)

        for imatr in range(self.number_of_scan):
            popt, pcov = curve_fit(funzione_raman2, data_cut_x, data_cut_y[:,imatr], p0 = [300, 1])#, bounds=([295, -np.inf],[self.temp_max, np.inf]))
            y_fit[:,imatr] = funzione_raman2(x_fit, popt[0], popt[1])
            T1[imatr] = popt[0]
            P1[imatr] = popt[1]

            residual = data_cut_y[:,imatr] - funzione_raman2(data_cut_x,*popt)
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((data_cut_y[:,imatr] - np.mean(data_cut_y[:,imatr]))**2)
            if ss_tot == 0:
                ss_tot = 1
                ss_res = 1
            r2[imatr] = 1- (ss_res/ss_tot)

            ############################################################
            # p = len(popt)
            # n = len(data_cut_x)
            alpha = 0.05 #95% confidence interval
            dof = max(0, len(data_cut_x) - len(popt)) #degree of freedom
            tval = tstud.ppf(1.0 - alpha/2., dof) #t-student value for the dof and confidence level
            sigma = np.diag(pcov)**0.5
            ET1[imatr] = sigma[0]*tval
            EP1[imatr] = sigma[1]*tval
            ########################################################

        if self.salva == 'yes':
            ds().nuova_fig(indice_fig=3)
            ds().titoli(xtag='nm', ytag='ratio (log)', titolo='')
            for i in range(self.number_of_scan):
                ds().dati(data_cut_x, data_cut_y[:,i], colore= palette[i])
                ds().dati(x_fit, y_fit[:,i], colore=palette[i])
            st.pyplot()

            ds().nuova_fig(indice_fig=2, indice_subplot = 326)
            ds().titoli(xtag='nm', ytag='ratio (log)', titolo='')
            for i in range(self.number_of_scan):
                ds().dati(data_cut_x, data_cut_y[:,i], colore= palette[i])
                ds().dati(x_fit, y_fit[:,i], colore=palette[i])
        return T1, r2, P1, ET1, EP1


    def average_over_wavelength(self, xx, yy, AS_min, AS_max, S_min, S_max):
        num_medie = 20
        media = np.zeros((num_medie, round(self.number_of_scan)))
        media_x = np.zeros(num_medie)
        step = 0
        delta = 2
        for j in range(num_medie):
            conta = 0
            for i in range(len(xx)):
                if xx[i]>AS_min + step and xx[i]<AS_min + step + delta and xx[i]<AS_max:
                    media_x[j] = xx[i] + media_x[j]
                    media[j, :] = yy[i, :] + media[j, :]
                    conta = conta + 1
            media[j, :] = media[j, :]/conta
            media_x[j] = media_x[j]/conta
            step = step + delta
        media_x, media = sm(media).nanremoval(media_x)
        return media_x, media


    def ratio_vs_power(self, power, AS_min = 588, AS_max = 616, S_min = 645, S_max = 699, T_RT = 295, laser_type = 633, selected_scan = 0):
        def funzione_log(ratio, P1, P_ref, wave):
            h_bar = 4.1356*1e-15
            kb = 8.617*1e-5
            c = 299792458
            l_laser = laser_type*1e-9
            wave = wave*1e-9

            numeratore   = ((h_bar*c)/kb)*((1/l_laser)-(1/wave))
            denominatore = np.log(ratio) - np.log(P1/P_ref) + ((h_bar*c)/(kb*T_RT))*((1/l_laser)-(1/wave))
            return numeratore/denominatore

        data_cut_x, data_cut_y = self.cut_data_optimal_range(AS_min, AS_max, S_min, S_max)
        media_x, media = self.average_over_wavelength(data_cut_x, data_cut_y, AS_min, AS_max, S_min, S_max)

        if self.salva == 'yes':
            ds().nuova_fig(indice_fig=7)
            ds().titoli(xtag='nm', ytag='ratio', titolo='')
            for i in range(self.number_of_scan):
                for j in range(len(media[:,0])):
                    ds().dati(media_x[j], media[j,i], colore= palette[j], scat_plot ='scat', larghezza_riga =15)
            st.pyplot()

            # ds().nuova_fig(indice_fig=2, indice_subplot = 324)
            # ds().titoli(xtag='nm', ytag='ratio', titolo='')
            # for i in range(self.number_of_scan):
            #     for j in range(len(media[:,0])):
            #         ds().dati(media_x[j], media[j,i], colore= palette[j], scat_plot ='scat', larghezza_riga =15)

            # st.write('temperature growth')
            # x = np.array(power, dtype="float")
            # ds().nuova_fig(indice_fig=8)
            # ds().titoli(xtag='I [mW/um^2]', ytag='ratio', titolo='')
            # for j in range(len(media[:,0])):
            #     ds().dati(x[:], media[j,:], colore= palette[j], scat_plot ='scat', larghezza_riga =15, descrizione=str(round(media_x[j],1)))
            # st.pyplot()

            # ds().nuova_fig(indice_fig=2, indice_subplot = 326)
            # ds().titoli(xtag='I [mW/um^2]', ytag='ratio', titolo='')
            # for j in range(len(media[:,0])):
            #     ds().dati(x[:], media[j,:], colore= palette[j], scat_plot ='scat', larghezza_riga =15, descrizione=str(round(media_x[j],1)))


        range_of_wavelength = pd.DataFrame()
        for i, wave in enumerate(media_x):
            vet_temp = []
            for j, potenza in enumerate(power):
                vet_temp.append(funzione_log(media[i, j], potenza, power[self.number_of_scan-1-selected_scan], wave))
            range_of_wavelength[wave] = vet_temp
        range_of_wavelength.index = power

        medie_temperature = range_of_wavelength.mean(axis = 1)
        sigma_temperature = range_of_wavelength.std(axis = 1)

        if self.salva == 'yes':
            st.write('Temperature density of states')
            ds().nuova_fig(20)
            # ds().titoli(xtag='I [mW/um^2]', ytag='T [k]', titolo='')
            ds().titoli(xtag='P [uW]', ytag='T [k]', titolo='')
            for j, col in enumerate(range_of_wavelength.columns):
                ds().dati(range_of_wavelength.index.tolist(), range_of_wavelength[col], colore = palette[j], scat_plot = 'scat', larghezza_riga = 15, descrizione=str(round(col)))
            ds().legenda()
            st.pyplot()

        return medie_temperature, sigma_temperature


    def direct_standard(self, power, AS_min = 588, AS_max = 616, S_min = 645, S_max = 699, T_RT = 295, laser_type = 633, selected_scan = 0):
        def funzione_direct(ratio, P1, P_ref, wave):
            h_bar = 4.1356*1e-15
            kb = 8.617*1e-5
            c = 299792458
            l_laser = laser_type*1e-9
            wave = wave*1e-9

            numeratore   = (h_bar*c/kb)*((1/wave)-(1/l_laser))
            denominatore = np.log((P1/P_ref)*(1/ratio)*(np.exp(((h_bar*c)/(kb*T_RT))*(1/(wave) - (1/l_laser)))-1)+1)
            return numeratore/denominatore

        data_cut_x, data_cut_y = self.cut_data_optimal_range(AS_min, AS_max, S_min, S_max)
        media_x, media = self.average_over_wavelength(data_cut_x, data_cut_y, AS_min, AS_max, S_min, S_max)

        if self.salva == 'yes':
            ds().nuova_fig(indice_fig=7)
            ds().titoli(xtag='nm', ytag='ratio', titolo='')
            for i in range(self.number_of_scan):
                for j in range(len(media[:,0])):
                    ds().dati(media_x[j], media[j,i], colore= palette[j], scat_plot ='scat', larghezza_riga =15)
            st.pyplot()

        range_of_wavelength = pd.DataFrame()
        for i, wave in enumerate(media_x):
            vet_temp = []
            for j, potenza in enumerate(power):
                vet_temp.append(funzione_direct(media[i, j], potenza, power[self.number_of_scan-1-selected_scan], wave))
            range_of_wavelength[wave] = vet_temp
        range_of_wavelength.index = power

        medie_temperature = range_of_wavelength.mean(axis = 1)
        sigma_temperature = range_of_wavelength.std(axis = 1)

        if self.salva == 'yes':
            st.write('Temperature density of states')
            ds().nuova_fig(20)
            ds().titoli(xtag='P [uW]', ytag='T [k]', titolo='')
            for j, col in enumerate(range_of_wavelength.columns):
                ds().dati(range_of_wavelength.index.tolist(), range_of_wavelength[col], colore = palette[j], scat_plot = 'scat', larghezza_riga = 15, descrizione=str(round(col)))
            ds().legenda()
            st.pyplot()

        return medie_temperature, sigma_temperature



        # ██ ███    ██ ████████ ███████ ███    ██ ███████ ██ ████████ ██    ██     ███    ███  █████  ██████
        # ██ ████   ██    ██    ██      ████   ██ ██      ██    ██     ██  ██      ████  ████ ██   ██ ██   ██
        # ██ ██ ██  ██    ██    █████   ██ ██  ██ ███████ ██    ██      ████       ██ ████ ██ ███████ ██████
        # ██ ██  ██ ██    ██    ██      ██  ██ ██      ██ ██    ██       ██        ██  ██  ██ ██   ██ ██
        # ██ ██   ████    ██    ███████ ██   ████ ███████ ██    ██       ██        ██      ██ ██   ██ ██


class intensity_map:

    def __init__(self, salva):
        self.salva = salva

    def eval_power(self, xy_mesh, intensity_sp, laser_power):
        if self.salva == 'yes':
            st.write('evaluating the real power on the particle')

            xx, yy = xy_mesh

            ds().nuova_fig(indice_fig=2, plot_dimension = '3d', indice_subplot = 322)
            ds().titoli(titolo='original with fit')
            ds().dati(xx, yy, scat_plot ='3D', z = intensity_sp)

            ds().nuova_fig(indice_fig=9, plot_dimension = '3d')
            ds().titoli(titolo='original with fit')
            ds().dati(xx, yy, scat_plot ='3D', z = intensity_sp)
            st.pyplot()


        tot_intensity = sum(sum(intensity_sp))
        intensity_sp_real = np.zeros((int(len(intensity_sp[:,0])), int(len(intensity_sp[0,:]))))
        intensity_sp_real = intensity_sp*(laser_power/tot_intensity)*1000# the 1000 is in place to convert into uW
        return intensity_sp_real

    def fit_guass(self, x, y, xy_mesh, xc, yc, amp, intensity_sp, rad, laser_power, punti_cross_section_geometrica = 10):
        def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y, theta):
            (x, y) = xy_mesh
            a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2);
            b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2);
            c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2);
            gauss = amp*np.exp( - (a*(x-xc)**2 + 2*b*(x-xc)*(y-yc) + c*(y-yc)**2));
            return np.ravel(gauss)

        # set initial parameters to build mock data
        sigma_x, sigma_y = 0.2, 0.2
        theta = 0
        guess_vals = [amp, xc, yc, sigma_x, sigma_y, theta]
        fit_params, cov_mat = curve_fit(gaussian_2d, xy_mesh, np.ravel(intensity_sp), p0=guess_vals)#, bounds=((amp-20,xc-0.015,yc-0.015,0.1,0.1,-np.inf), (amp+20,xc+0.015,yc+0.015,0.2,0.2,np.inf)))

        intensity_gauss = np.zeros((int(len(intensity_sp[:,0])), int(len(intensity_sp[0,:]))))
        for i in range(len(intensity_sp[:,0])):
            for j in range(len(intensity_sp[0,:])):
                intensity_gauss[i,j] = gaussian_2d((x[j], y[i]), fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5])

        xx, yy = xy_mesh

        if self.salva == 'yes':
            ds().nuova_fig(indice_fig=2, plot_dimension = '3d', indice_subplot = 322)
            ds().titoli(titolo='original with fit')
            ds().dati(xx, yy, scat_plot ='3D', z = intensity_sp)
            ds().dati(xx, yy, scat_plot ='3D_wire', z = intensity_gauss, colore='black')

            ds().nuova_fig(indice_fig=9, plot_dimension = '3d')
            ds().titoli(titolo='original with fit')
            ds().dati(xx, yy, scat_plot ='3D', z = intensity_sp)
            ds().dati(xx, yy, scat_plot ='3D_wire', z = intensity_gauss, colore='black')
            st.pyplot()

        def area_particle(x0 = 0, y0 = 0):
            intensity_sp_sum = 0

            x_rad = np.linspace(0,rad*2, punti_cross_section_geometrica)
            y_rad = np.linspace(0,rad*2, punti_cross_section_geometrica)

            for i in range(len(x_rad)):
                for j in range(len(y_rad)):
                    x_pos = x_rad[i] - rad + x0
                    y_pos = y_rad[j] - rad + y0
                    dist = np.sqrt((x0 - x_pos)**2 + (y0 - y_pos)**2)
                    if dist<rad:
                        valore = gaussian_2d((x_pos, y_pos), fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5])
                        intensity_sp_sum = intensity_sp_sum + valore
            # intensity_sp_sum = intensity_sp_sum*(2*rad/len(x_rad))*(2*rad/len(y_rad))
            intensity_sp_sum = intensity_sp_sum*(3.14*rad*rad/(len(x_rad)*len(y_rad)))
            return intensity_sp_sum

        if self.salva == 'yes':
            st.text('calculating the geometrical cross-section')

        intensity_sp_real = np.zeros((int(len(intensity_sp[:,0])), int(len(intensity_sp[0,:]))))
        for i in range(len(intensity_sp[:,0])):
            for j in range(len(intensity_sp[0,:])):
                intensity_sp_real[i,j] = area_particle(x[j], y[i])

        tot_intensity = sum(sum(intensity_sp_real))

        # intensity_sp = intensity_sp/10000

        # intensity_sp_real[:,:] = intensity_sp_real[:,:]*(laser_power/tot_intensity)
        intensity_sp_real[:,:] = intensity_sp_real[:,:]*(laser_power/tot_intensity)/(rad*rad*3.14)

        # intensity_sp_real[:,:] = intensity_sp_real[:,:]*(laser_power/sum(sum(intensity_sp)))/(rad*rad*3.14)
        # intensity_sp_real[:,:] = intensity_sp_real[:,:]*(laser_power/sum(sum(intensity_sp)))

        return intensity_sp_real
