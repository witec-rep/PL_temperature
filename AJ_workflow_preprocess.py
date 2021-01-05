from AJ_draw import disegna as ds
from AJ_function_4_analysis import size_manipulator as sm

from AJ_PL_analisi import normalization as nr
from AJ_PL_analisi import pulizzia as pl
from AJ_PL_analisi import data_storage
import numpy as np
import streamlit as st

palette = ['#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']

class preprocess:
    """
    :param file_lamp: file of the lamp Spectrum
    :param file_bkg_single_disks: file of the background for the dark field of the single disk
    :param file_bkg_laser: file of the background for the photoluminescence
    :param file_disk: file of the single disk dark field Spectrum
    :param file_PL: file of the photoluminescence spectrum
    :param num_of_scan: number of scan in the 'file_PL'
    """
    def __init__(self, raw_data, num_of_scan):
        self.raw_data = raw_data
        self.num_of_scan = num_of_scan

    def calc_deriv_thershold(self):
        raw_PL = self.raw_data
        y_deriv = np.zeros((len(raw_PL[0]), self.num_of_scan, self.num_of_scan))

        dx = raw_PL[0][1] - raw_PL[0][0]
        for i in range(self.num_of_scan):
            for j in range(self.num_of_scan):
                y_deriv[:,i,j] = np.gradient(raw_PL[1][:,i,j], dx)

        S_min = 640
        i_pl = 0
        for i in range(len(raw_PL[0])):
            if raw_PL[0][i] < S_min+1:
                i_pl = i

        ds().nuova_fig(indice_fig=1, indice_subplot =211, width =12)
        ds().titoli(titolo='', xtag="",ytag="Derivate",griglia=2)
        max_S = 0
        for i in range(self.num_of_scan):
            for j in range(self.num_of_scan):
                ds().dati(raw_PL[0], y_deriv[:,i,j], colore=palette[i])
                if max_S < max(y_deriv[i_pl:,i,j]):
                    max_S = max(y_deriv[i_pl:,i,j])
        ds().range_plot(bottomY = 0, topY =100+max_S)

        ds().nuova_fig(indice_fig=1, indice_subplot =212, width =12)
        ds().titoli(titolo='', xtag="",ytag="PL",griglia=2)
        max_S = 0
        for i in range(self.num_of_scan):
            for j in range(self.num_of_scan):
                ds().dati(raw_PL[0], raw_PL[1][:,i,j], colore=palette[i])
                if max_S < max(raw_PL[1][i_pl:,i,j]):
                    max_S = max(raw_PL[1][i_pl:,i,j])
        ds().range_plot(bottomY = 0, topY =100+max_S)
        st.pyplot()

    def remove_cosmic_ray(self, soglia_derivata):
        """
        calculated the heat map
        """
        def interpolazione(x, p1x, p1y, p2x, p2y):
            y = (((x - p2x)/(p1x - p2x))*(p1y - p2y)) + p2y
            return y

        raw_PL = self.raw_data
        y_deriv = np.zeros((len(raw_PL[0]), self.num_of_scan, self.num_of_scan))

        dx = raw_PL[0][1] - raw_PL[0][0]
        for i in range(self.num_of_scan):
            for j in range(self.num_of_scan):
                y_deriv[:,i,j] = np.gradient(raw_PL[1][:,i,j], dx)

        cut_val = soglia_derivata

        for i in range(self.num_of_scan):
            for j in range(self.num_of_scan):
                k=0
                for kk in range(len(raw_PL[0])):
                    if y_deriv[k,i,j] > cut_val:
                        raw_PL[1][k-1,i,j] = interpolazione(raw_PL[0][(k-1)], raw_PL[0][(k-1)-2], raw_PL[1][(k-1)-2,i,j], raw_PL[0][(k-1)+5], raw_PL[1][(k-1)+5,i,j])
                        raw_PL[1][k,i,j] = interpolazione(raw_PL[0][(k)],     raw_PL[0][(k)-2],   raw_PL[1][(k)-2,i,j],   raw_PL[0][(k)+5], raw_PL[1][(k)+4,i,j])
                        raw_PL[1][k+1,i,j] = interpolazione(raw_PL[0][(k+1)], raw_PL[0][(k+1)-2], raw_PL[1][(k+1)-2,i,j], raw_PL[0][(k+1)+5], raw_PL[1][(k+1)+5,i,j])
                        raw_PL[1][k+2,i,j] = interpolazione(raw_PL[0][(k+2)], raw_PL[0][(k+2)-2], raw_PL[1][(k+2)-2,i,j], raw_PL[0][(k+2)+5], raw_PL[1][(k+2)+5,i,j])
                        raw_PL[1][k+3,i,j] = interpolazione(raw_PL[0][(k+3)], raw_PL[0][(k+3)-2], raw_PL[1][(k+3)-2,i,j], raw_PL[0][(k+3)+5], raw_PL[1][(k+3)+5,i,j])
                        k = k + 5
                    k = k + 1
                    if k >= len(raw_PL[0])-20:
                        break

        data_x = np.zeros(len(raw_PL[0]))
        data_y = np.zeros((len(raw_PL[1][:,0,0]), self.num_of_scan, self.num_of_scan))

        data_x = raw_PL[0]
        data_y[:,:,:] = raw_PL[1][:,:,:]

        return data_x, data_y
