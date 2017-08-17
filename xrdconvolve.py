# -*- coding: latin-1 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz, erfc
from scipy.optimize import fsolve
import os.path
this_dir, this_filename = os.path.split( __file__ )
DATA_PATH = os.path.join( this_dir, "data" )
import xrdsim, xrdanal

def sim_std( dict_phases, phase_id, instrum_data, raw_data, ratio_alpha = 0.52, strain = 0, alpha3 = False, affich = 0 ):
	"""

	Simule le spectre d'un échantillon standard (pas de déformation ou d'influence de la taille des cristaux)

	Simule les pics K-alpha 1 et K-alpha 2 du spectre du cuivre

	"""

	[mat, emetteur, raie] = dict_phases['MatBase']
	[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = xrdanal.read_melange( mat, emetteur, raie, affich=0 )
	lam_1 = 1.540562
	lam_2 = 1.544390
	lam_3 = 1.53416
	phase_data = dict_phases[ phase_id ]
	I_max_obs = max( raw_data.count )
	I_max_calc = 0
	[dth0, s, K1, K2]  = instrum_data.corr_th
	sol = instrum_data.delta_sol
	h = instrum_data.mask/2.
	[U, V, W] = instrum_data.calib_FWHM
	[b, c] = instrum_data.calib_ff
	R_diff = instrum_data.R_diff
	DS = instrum_data.DS

	theta_range = raw_data.theta
	count_range = raw_data.count*0

	for i in range( len( phase_data[5] ) ):
		theta_calc = phase_data[5][i][3][0] #1theta(rad)
		R = phase_data[5][i][8][0]

		theta_obs = (2*theta_calc + dth0 - 2*s*np.cos( theta_calc )/R_diff - K1*DS**2/6/np.tan( theta_calc ) - K2*(sol**2/12 + h**2/3/R_diff**2)/np.tan( theta_calc*2 ) )*360/2/np.pi #2theta
		FWHM = (U*(np.tan(theta_obs*np.pi/360))**2 + V*np.tan(theta_obs*np.pi/360) + W)**0.5 * 360/2/np.pi
		ff = b*theta_obs + c

		A = R
		beta = FWHM/ff
		I0 = A/beta

		f_k = lambda k : 2*(1 + k**2)**0.5*erfc(k)/np.pi**0.5/np.exp(-k**2)-ff
		k = fsolve( f_k, np.pi**-0.5 )

		#Fonction W*G
		beta_g = np.pi**0.5*FWHM/2/(1+k**2)**0.5
		beta_l = k*beta_g*np.pi**0.5
		I0lg = (I0/beta_l/(wofz(1j*k)).real)**0.5

		#Correction pour la déformation
		beta_strain = 4*strain*np.tan(theta_obs*np.pi/360)*360/2/np.pi
		I0lg = I0lg*( beta_l/(beta_l+beta_strain) )**0.5
		beta_l += beta_strain
		k = beta_l/np.pi**0.5/beta_g

		f_pic_1 = lambda theta: beta_l*I0lg**2*(wofz( np.pi**0.5*(theta - theta_obs)/beta_g + 1j*k )).real

		theta_2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(theta_obs*np.pi/360) )
		I0lg_2 = ratio_alpha**0.5*I0lg

		f_pic_2 = lambda theta: beta_l*I0lg_2**2*(wofz( np.pi**0.5*(theta - theta_2)/beta_g + 1j*k )).real

		for j in range( len( raw_data.theta ) ):
			theta = theta_range[j]
			count_range[j] += f_pic_1(theta) + f_pic_2(theta)

	I_max_calc = max( count_range )
	count_range = count_range/I_max_calc*I_max_obs

	if affich == 1:
		xrdsim.trace(raw_data)
		plt.plot(theta_range, count_range)
		plt.show()


	return theta_range, count_range		

