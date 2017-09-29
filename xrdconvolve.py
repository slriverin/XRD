# -*- coding: latin-1 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz, erfc
from scipy.optimize import fsolve, curve_fit
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d
from copy import deepcopy
import functools
import math
import os.path
this_dir, this_filename = os.path.split( __file__ )
DATA_PATH = os.path.join( this_dir, "data" )
import xrdsim, xrdanal
from XRD.XRD import funcs

f_D = funcs.f_D
intD = funcs.intD

try:
	input = raw_input
except:
	pass


def sim_std( dict_phases, phase_id, instrum_data, raw_data, ratio_alpha = 0.52, alpha3 = False, affich = 0, corr_asym = True ):
	"""

	Simule le spectre d'un échantillon standard

	Simule les pics K-alpha 1 et K-alpha 2 du spectre du cuivre

	"""

	[mat, emetteur, raie] = dict_phases['MatBase']
	[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = xrdanal.read_melange( mat, emetteur, raie, affich=0 )
	lam_1 = 1.540562
	lam_2 = 1.544390
	lam_3 = 1.53416
	phase_data = dict_phases[ phase_id ]
	cryst_struct = phase_data[1]
	[dth0_s, s_s, K1_s, K2_s]  = instrum_data.corr_th
	dth0 = dth0_s[0]
	s = s_s[0]
	K1 = K1_s[0]
	K2 = K2_s[0]
	sol = instrum_data.delta_sol
	h = instrum_data.mask/2.
	[U_s, V_s, W_s] = instrum_data.calib_FWHM
	U = U_s[0]
	V = V_s[0]
	W = W_s[0]
	[b_s, c_s] = instrum_data.calib_ff
	b = b_s[0]
	c = c_s[0]
	R_diff = instrum_data.R_diff
	DS = instrum_data.DS
	strain = phase_data[8][0]

	theta_range = raw_data.theta
	count_range = raw_data.count*0
	liste_f = []
	if affich==1:
		print('\tPhase : ' + phase_id )

	for i in range( len( phase_data[5] ) ):
		if affich==1:
			print('\t\tPlan : ' + str(phase_data[5][i][0]))
		theta_calc = phase_data[5][i][3][0] #1theta(rad)
		R = phase_data[5][i][8][0]
		FV = phase_data[7][0] #Fraction volumique

		theta_obs = (2*theta_calc + dth0 - 2*s*np.cos( theta_calc )/R_diff - K1*DS**2/6/np.tan( theta_calc ) - K2*sol**2/6/np.tan( theta_calc*2 ) )*360/2/np.pi #2theta
		FWHM = (U*(np.tan(theta_obs*np.pi/360))**2 + V*np.tan(theta_obs*np.pi/360) + W)**0.5 * 360/2/np.pi
		ff = b*theta_obs + c

		#Correction pour orientation préférentielle
		if cryst_struct == 'BCC' or cryst_struct == 'FCC' or cryst_struct == 'DC':
			h, k, l = phase_data[5][i][0]
			l_N = (float(h)**2 + float(k)**2 + float(l)**2)**0.5
			x = float(h) / l_N
			y = float(k) / l_N
			z = float(l) / l_N
			K41 = 2.5*(x**4 + y**4 + z**4) - 1.5
			C41 = phase_data[9]['C410'][0]
			K61 = 21.65625*(x**6 + y**6 + z**6) - 29.53125*(x**4 + y**4 + z**4) + 8.4375
			C61 = phase_data[9]['C610'][0]
			K81 = 1./3*(97.5*(x**8 + y**8 + z**8) - 182.*(x**6 + y**6 + z**6) + 105.*(x**4 + y**4 + z**4) - 17.5)
			C81 = phase_data[9]['C810'][0]

			text_fact = 1. + C41*K41 + C61*K61 + C81*K81 

		A = R * FV * text_fact
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

		#def f_pic_1(theta):
		#	return beta_l*I0lg**2*(wofz( np.pi**0.5*(theta - theta_obs)/beta_g + 1j*k )).real
		f_pic_1 = funcs.def_f_pic( I0lg, theta_obs, beta_g, beta_l )

		theta_2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(theta_obs*np.pi/360) )
		I0lg_2 = ratio_alpha**0.5*I0lg

		f_pic_2 = funcs.def_f_pic( I0lg_2, theta_2, beta_g, beta_l )

		count_temp = np.zeros(len(raw_data.theta))
		count_temp_2 = np.zeros(len(raw_data.theta))

		PBratio = np.max(raw_data.count)/np.min(raw_data.count)

		#Correction pour asymétrie
		if corr_asym == True :
			S = instrum_data.mask
			H = 24.0
			L=instrum_data.R_diff
			phi_min = np.arccos( np.cos(theta_obs*2*np.pi/360)*(((H+S)/L)**2+1)**0.5 )*360/2/np.pi
			phi_infl = np.arccos( np.cos(theta_obs*2*np.pi/360)*(((H-S)/L)**2+1)**0.5 )*360/2/np.pi
			cutoff=0.1/PBratio*f_pic_1(theta_obs)
			if theta_obs < 90:
				surf_D = intD(theta_obs, phi_min, phi_infl, H, S, L)
				for j in range( len( raw_data.theta) ):
					phi = theta_range[j]
					if f_pic_1(phi) < cutoff:
						continue
					fd = lambda t : f_D(t, theta_obs, phi_min, phi_infl, H, S, L)
					f_asym = lambda t: fd(t)*f_pic_1(theta_obs+phi-t)/surf_D
					count_temp[j] = quad(f_asym, phi_min, theta_obs)[0]
					
			else:
				surf_D = intD(180. - theta_obs, 180. - phi_min, 180. - phi_infl, H, S, L)
				for j in range( len( raw_data.theta) ):
					phi = theta_range[j]
					if f_pic_1(phi) < cutoff:
						continue
					fd = lambda t : f_D(t, 180. - theta_obs, 180. - phi_min, 180. - phi_infl, H, S, L)
					f_asym = lambda t: fd(180.-t)*f_pic_1(theta_obs+phi-t)/surf_D
					count_temp[j] = quad(f_asym, theta_obs, phi_min)[0]

			#pic alpha2:
			phi_min_2 = np.arccos( np.cos(theta_2*2*np.pi/360)*(((H+S)/L)**2+1)**0.5 )*360/2/np.pi
			phi_infl_2 = np.arccos( np.cos(theta_2*2*np.pi/360)*(((H-S)/L)**2+1)**0.5 )*360/2/np.pi
			surf_D_2 = intD(theta_2, phi_min_2, phi_infl_2, H, S, L)
			cutoff=0.1/PBratio*f_pic_2(theta_2)
			if theta_2 < 90:
				surf_D_2 = intD(theta_2, phi_min_2, phi_infl_2, H, S, L)
				for j in range( len( raw_data.theta) ):
					phi = theta_range[j]
					if f_pic_2(phi) < cutoff:
						continue
					fd = lambda t : f_D(t, theta_2, phi_min_2, phi_infl_2, H, S, L)
					f_asym = lambda t: fd(t)*f_pic_2(theta_2+phi-t)/surf_D_2
					count_temp_2[j] = quad(f_asym, phi_min_2, theta_2)[0]
			else:
				surf_D_2 = intD(180. - theta_2, 180. - phi_min_2, 180. - phi_infl_2, H, S, L)
				for j in range( len( raw_data.theta) ):
					phi = theta_range[j]
					if f_pic_2(phi) < cutoff:
						continue
					fd = lambda t : f_D(t, 180. - theta_2, 180. - phi_min_2, 180. - phi_infl_2, H, S, L)
					f_asym = lambda t: fd(180.-t)*f_pic_2(theta_2+phi-t)/surf_D_2
					count_temp_2[j] = quad(f_asym, theta_2, phi_min_2)[0]

		else:
			for j in range( len( raw_data.theta ) ):
				theta = theta_range[j]
				count_temp[j] = f_pic_1(theta)
				count_temp_2[j] = f_pic_2(theta)

		liste_f.append(count_temp)
		liste_f.append(count_temp_2)
		count_range += count_temp + count_temp_2		

	if affich == 1:
		xrdsim.trace(raw_data)
		plt.plot(theta_range, count_range)
		plt.show()


	return theta_range, count_range, liste_f		

def rietvelt_init( spectre, dict_phases, instrum_data, corr_asym = True ):
	"""

	Initie l'analyse de Rietvelt

	Args :
		spectre
		dict_phases
		instrum_data

	Conditions :
		spectre doit contenir des données brutes ainsi que les données d'arrière-plan
		Les pics doivent être identifiés
		Si spectre.rietvelt = True, un dialogue demande si on veut remettre à zéro

	Extrant : structure spectre avec nouvelles sous-structures :

		spectre.etat.rietvelt = True	: Booléen faux par défaut 
		spectre.rietvelt_calc : Approxime la solution selon les paramètres donnés
		spectre.rietvelt_iter : itère la solution de Rietvelt
		spectre.fit.dict_args : Dictionnaire de tous les arguments raffinables
		spectre.fit.dict_args[arg] = [ actif, [lim_min, lim_max] ]
		spectre.dict_phases
		spectre.instrum_data

		arg :
			'pxxy' (ex. 'p123') : Arguments de la liste de pics (ex. 3e argument, e.g. th0, du pic 12). Non utilisé pour le raffinement de Rietvelt
			'bx' : Ordonnée du xe point d'arrière-plan
			'p:phase_id:x'  : Fraction volumique de la phase 'phase_id'
			'p:phase_id:a'  : Paramètre de maille 'a' de la phase 'phase_id'
			'p:phase_id:c'  : Paramètre de maille 'c' de la phase 'phase_id'
			'p:phase_id:s'  : Déformation de la phase 'phase_id'
			'p:phase_id:Cijp' : Paramètre de correction pour orientation préférentielle
			'i:dth:c'	: Correction zéro
			'i:dth:s'	: Déplacement échantillon
			'i:dth:1'	: Facteur K1 (spécimen plat)
			'i:dth:2'	: Facteur K2 (divergence axiale)
			'i:ra:ra'		: Ratio d'intensité alpha2/alpha1
			'i:FWHM:U'	: Paramètres régression FWHM
			'i:FWHM:V'
			'i:FWHM:W'
			'i:ff:b'	: Paramètres régression form factor
			'i:ff:c'

		actif : Interrupteur Vrai/Faux déterminant si le paramètre est raffiné ou pas
		lim_min, lim_max : Limites inférieure et supérieure des valeurs acceptables pour le paramètre

	"""

	if spectre.etat.rietvelt == True :
		print( u'Régression de Rietvelt déjà existante. Écraser?' )
		choix = input( '1 : Oui; autre : non. ... : ' )
		if choix != '1':
			return

	if spectre.etat.raw_data == False or spectre.etat.back_list == False or spectre.etat.peak_list == False:
		print( u'Le spectre doit avoir des données brutes, une liste de pics et une liste de points d\'arrière-plan' )
		return
	
	spectre.etat.rietvelt = True
	spectre.etat.reg = True
	spectre.etat.rietvelt_opt = False
	spectre.etat.corr_asym = corr_asym
	spectre.dict_phases = deepcopy(dict_phases)
	spectre.instrum_data = deepcopy(instrum_data)
	spectre.dict_args = {}

	#Construction du dictionnaire des arguments
	#1. Points d'arrière-plan
	for i in range( len( spectre.back_pts_list ) ):
		spectre.dict_args[ 'b' + str(i) ] = [ True, [0., np.inf] ]

	#2. Informations sur chacune des phases
	xmax = 0.
	for phase in spectre.dict_phases:
		if phase == 'MatBase':
			continue

		cryst_struct = spectre.dict_phases[phase][1]
		x = spectre.dict_phases[phase][7][0]

		if x > xmax:
			xmax = x
			phasemax = phase
		
		spectre.dict_args[ 'p:' + phase + ':x' ] = [ True, [0., 1.] ]
		spectre.dict_args[ 'p:' + phase + ':a' ] = [ True, [1., 10.] ]
		if cryst_struct == 'BCT' or cryst_struct == 'HCP':
			spectre.dict_args[ 'p:' + phase + ':c' ] = [ False, [1., 10.] ]
		else:	
			spectre.dict_args[ 'p:' + phase + ':c' ] = [ False, [1., 10.] ]
		spectre.dict_args[ 'p:' + phase + ':s' ] = [ True, [0., 1.] ]

		for fact in spectre.dict_phases[phase][9]:
			spectre.dict_args[ 'p:' + phase + ':' + fact ] = [ False, [-np.inf, np.inf] ]

	spectre.dict_args[ 'p:' + phasemax + ':x' ] = [ False, 'B' ]

	#3. Facteurs de correction
	spectre.dict_args[ 'i:dth:c' ] = [ False, [-1.e-3, 1.e-3] ]
	spectre.dict_args[ 'i:dth:s' ] = [ False, [-0.5, 0.5] ]
	spectre.dict_args[ 'i:dth:1' ] = [ False, [0., 10.] ]
	spectre.dict_args[ 'i:dth:2' ] = [ False, [0., 10.] ]
	spectre.dict_args[ 'i:ra:ra' ] = [ False, [0.4, 0.6] ]
	spectre.dict_args[ 'i:FWHM:U' ] = [ False, [-np.inf, np.inf] ]
	spectre.dict_args[ 'i:FWHM:V' ] = [ False, [-np.inf, np.inf] ]
	spectre.dict_args[ 'i:FWHM:W' ] = [ False, [-np.inf, np.inf] ]
	spectre.dict_args[ 'i:ff:b' ] = [ False, [-np.inf, np.inf] ]
	spectre.dict_args[ 'i:ff:c' ] = [ False, [-np.inf, np.inf] ]
	

	spectre.rietvelt_calc = rietvelt_calc.__get__( spectre, spectre.__class__ )
	spectre.rietvelt_opt = rietvelt_opt.__get__( spectre, spectre.__class__ )
	spectre.rietvelt_params = rietvelt_params.__get__( spectre, spectre.__class__ )

	spectre.background_approx()
	return spectre

def rietvelt_calc( self, affich = 0 ):
	"""

	Calcule la solution approximative à partir des paramètres actuels

	Args :
		affich : mettre à 1 pour afficher le résultat

	Met à jour la structure 'spectre' avec les données approximées et les statistiques de convergence

	"""

	theta_range = self.raw_data.theta
	count_range = self.raw_data.count*0
	self.fit.data_reg = xrdsim.data()
	self.fit.data_reg.theta = theta_range
	self.fit.data_reg.count = count_range*0.
	self.fit.residu = xrdsim.data()
	self.fit.residu.theta = theta_range
	self.fit.residu.count = count_range*0.
	self.background_approx()
	liste_psf = []

	for phase_id in self.dict_phases:
		if phase_id == 'MatBase':
			continue

		A = sim_std( self.dict_phases, phase_id, self.instrum_data, self.raw_data, self.instrum_data.ratio_alpha, corr_asym = self.etat.corr_asym, affich=0 )
		count_range += A[1]
		for i in range( len( A[2] ) ):
			liste_psf.append( A[2][i] )

	A_back = simps( self.data_back.count, self.data_back.theta )
	A_obs = simps( self.raw_data.count, self.raw_data.theta )
	A_simul = simps( count_range, theta_range )

	ratiob = A_back/A_obs

	count_range += self.data_back.count / A_back * A_simul * ratiob/(1-ratiob)

	A_simul = simps( count_range, theta_range )
	count_range = count_range * A_obs/A_simul
	self.fit.data_reg.count = count_range


	s_res_2 = 0.
	s_iobs_2 = 0.
	s_w_iobs_2 = 0.
	s_abs_res = 0.
	s_iobs = 0.
	s_w_res_2 = 0.

	for i in range( len( self.fit.data_reg.theta ) ):
		i_obs = self.raw_data.count[i]
		i_calc = self.fit.data_reg.count[i]
		w = 1./max(i_obs, 1.)
		res = i_obs - i_calc

		self.fit.residu.count[i] = res

		#Pour calcul de R 
		s_res_2 += res**2
		s_iobs_2 += i_obs**2

		#Pour calcul R_p 
		s_abs_res += np.abs( res )
		s_iobs += i_obs

		#Pour calcul R_wp
		s_w_res_2 += w*res**2
		s_w_iobs_2 += w*i_obs**2

	self.fit.R = 100.*( s_res_2/s_iobs_2 )**0.5	#[Howard1989]
	self.fit.R_p = 100.*s_abs_res/s_iobs		#[Parrish2004]
	self.fit.R_wp = 100.*( s_w_res_2/s_w_iobs_2 )**0.5#[Parrish2004]
	self.fit.s_w_iobs_2 = s_w_iobs_2
	self.fit.s_w_res_2 = s_w_res_2
	self.est_noise()
	
	if self.etat.rietvelt_opt == True:
		m = len(self.fit.list_args)
		self.fit.GOF = ((self.fit.R_wp/100.)**2 * self.fit.s_w_iobs_2 / (len(self.raw_data.count) - m))**0.5
		
		S2 = 0.
		S1 = 0.
		s1 = []
		s2 = []
		z = []
		for i in range( len( self.raw_data.theta ) ):
			ai = self.fit.residu.count[i]/(self.raw_data.count[i])**0.5
			if i == 0:
				zi = 0.
			elif self.fit.residu.count[i]*self.fit.residu.count[i-1] < 0.:
				zi = 0.
			else:
				ai_moins = self.fit.residu.count[i-1]/(self.raw_data.count[i-1])**0.5
				zi = ( 2*(ai**2 + ai_moins**2))**0.5 / (2 + (2*(ai**2 + ai_moins**2))**0.5 ) 

			
			S2 += (1 - zi**2)*ai**2
			S1 += zi*ai
			if zi == 0.:
				S2 += S1**2
				S1 = 0.
			s1.append(S1)
			s2.append(S2)
			z.append(zi)

		self.fit.fact_corr = S2 / self.fit.s_w_res_2
	
	else:
		self.fit.GOF = 0.
		self.fit.fact_corr = 0.

	if affich == 1:
		print( 'R = \t' + str( np.round( self.fit.R, 3 ) ) + ' %' )
		print( 'R_p = \t' + str( np.round( self.fit.R_p, 3 ) ) + ' %' )
		print( 'R_wp = \t' + str( np.round( self.fit.R_wp, 3) ) + ' %' )
		print( 'GOF =\t%.4f' %(self.fit.GOF) )
		print( 'Facteur de correction variance : %.4f' %(self.fit.fact_corr) )
		self.rietvelt_params()

		f, axarr = plt.subplots( 2, sharex = True )
		axarr[0].errorbar( self.raw_data.theta, self.raw_data.count, yerr = self.raw_data.count**0.5, color = 'k', fmt = '.', fillstyle = 'none', label = r'Donn\'{e}es brutes' )
		axarr[0].plot( theta_range, count_range, 'b', label = 'Rietveld' )
		axarr[0].plot( [], [], 'y', label = r'Pics s\'{e}par\'{e}s' )
		for i in range( len( liste_psf ) ):
			axarr[0].plot( theta_range, liste_psf[i]*A_obs/A_simul, 'y' )
		axarr[0].plot( self.data_back.theta, self.data_back.count, 'c', label = 'Background' )	
		axarr[0].set_ylabel( r'$I$ (comptes)' )
		axarr[0].legend()
		
		axarr[1].plot( self.raw_data.theta, self.fit.residu.count, label = r'R\'{e}sidu' )
		axarr[1].plot( [self.raw_data.theta[0], self.raw_data.theta[-1]], [0., 0.], 'k:' )
		axarr[1].set_xlabel( r'$2 \theta (^o)$' )
		axarr[1].set_ylabel( r'$I$ (comptes' )
		axarr[1].legend()
		plt.show()
		

def rietvelt_opt( self, affich = 1 ):
	"""

	Calcule les paramètres de régression par la méthode de Rietvelt.

	Une variable "dummy" contenant une copie du spectre est créée. Une fonction dont les paramètres
	sont 'theta' et les paramètres à optimiser est générée et les paramètres optimaux sont trouvés à 
	l'aide de la fonction scipy.optimize.curve_fit.

	La solution trouvée est ensuite transférée dans le spectre initial et la fonction 'rietvelt_calc'
	est appelée pour modéliser la solution.

	Args :
		affich : Mettre à 1 pour afficher les résultats

	"""	

	#Construction de la liste des arguments
	list_args = []
	params = []
	sigma = []
	p0 = []
	bound_min = []
	bound_max = []

	for arg in self.dict_args:
		if self.dict_args[arg][0] == True:
			list_args.append(arg)
			bound_min.append( self.dict_args[arg][1][0] )
			bound_max.append( self.dict_args[arg][1][1] )
			try:
				arg1, arg2, arg3 = arg.split(':')
			except:
				arg1 = ''
				arg2 = ''
				arg3 = ''
					
			if arg[0] == 'b':
				ptno = int( arg[1:] )
				p0.append( self.back_pts_list[ptno][1] )
			elif arg1 == 'p':
				phase_id = arg2
				phase = self.dict_phases[phase_id]
				if arg3 == 'x':
					p0.append( phase[7][0] )
				elif arg3 == 'a':
					p0.append( phase[2][0] )
				elif arg3 == 'c':
					c = phase[3][0]
					p0.append( phase[3][0] )
				elif arg3 == 's':
					p0.append( phase[8][0] )
				elif arg3[0] == 'C':
					p0.append( phase[9][arg3][0] )
			elif arg == 'i:ra:ra':
				p0.append( self.instrum_data.ratio_alpha )
			elif arg[0:5] == 'i:dth':
				if arg[-1] == 'c':
					p0.append( self.instrum_data.corr_th[0][0] )
				elif arg[-1] == 's':
					p0.append( self.instrum_data.corr_th[1][0] )
				elif arg[-1] == '1':
					p0.append( self.instrum_data.corr_th[2][0] )
				elif arg[-1] == '2':
					p0.append( self.instrum_data.corr_th[3][0] )
			elif arg[0:6] == 'i:FWHM':
				if arg[-1] == 'U':
					p0.append( self.instrum_data.calib_FWHM[0][0] )
				elif arg[-1] == 'V':
					p0.append( self.instrum_data.calib_FWHM[1][0] )
				elif arg[-1] == 'W':
					p0.append( self.instrum_data.calib_FWHM[2][0] )
			elif arg[0:4] == 'i:ff':
				if arg[-1] == 'b':
					p0.append( self.instrum_data.calib_ff[0][0] )
				elif arg[-1] == 'c':
					p0.append( self.instrum_data.calib_ff[1][0] )

	#Génération de la fonction à optimiser					
	sp_dummy = deepcopy( self )

	def f( theta, *args ):
		#update sp_dummy avec les arguments

		for i in range( len( args ) ):
			arg = list_args[i]
			try:
				arg1, arg2, arg3 = arg.split(':')
			except:
				arg1 = ''
				arg2 = ''
				arg3 = ''

			if arg[0] == 'b':
				ptno = int( arg[-1] )
				sp_dummy.back_pts_list[ptno][1] = args[i]
			elif arg1 == 'p':
				phase_id = arg2
				phase = sp_dummy.dict_phases[phase_id]
				if arg3 == 'x':
					phase[7] = (args[i], 0)
				elif arg3 == 'a':
					phase[2] = (args[i], 0)
				elif arg3 == 'c':
					phase[3] = (args[i], 0)
				elif arg3 == 's':
					phase[8] = (args[i], 0)
				elif arg3[0] == 'C':
					phase[9][arg3] = (args[i], 0)
			elif arg == 'i:ra:ra':
				sp_dummy.instrum_data.ratio_alpha = args[i]
			elif arg[0:5] == 'i:dth':
				if arg[-1] == 'c':
					sp_dummy.instrum_data.corr_th[0] = (args[i], 0)
				elif arg[-1] == 's':
					sp_dummy.instrum_data.corr_th[1] = (args[i], 0)
				elif arg[-1] == '1':
					sp_dummy.instrum_data.corr_th[2] = (args[i], 0)
				elif arg[-1] == '2':
					sp_dummy.instrum_data.corr_th[3] = (args[i], 0)
			elif arg[0:6] == 'i:FWHM':
				if arg[-1] == 'U':
					sp_dummy.instrum_data.calib_FWHM[0] = (args[i], 0)
				elif arg[-1] == 'V':
					sp_dummy.instrum_data.calib_FWHM[1] = (args[i], 0)
				elif arg[-1] == 'W':
					sp_dummy.instrum_data.calib_FWHM[2] = (args[i], 0)
			elif arg[0:4] == 'i:ff':
				if arg[-1] == 'b':
					sp_dummy.instrum_data.calib_ff[0] = (args[i], 0)
				elif arg[-1] == 'c':
					sp_dummy.instrum_data.calib_ff[1] = (args[i], 0)

		FV_tot = 0
		for cle in sp_dummy.dict_args:
			if sp_dummy.dict_args[cle][1] == 'B':
				phasemax = cle[2:-2]

		for phase_id in sp_dummy.dict_phases:
			if phase_id == 'MatBase' or phase_id == phasemax:
				continue
			FV_tot += sp_dummy.dict_phases[phase_id][7][0]


		sp_dummy.dict_phases[ phasemax ][7] = ( 1 - FV_tot, 0 )

		for phase_id in sp_dummy.dict_phases:
			if phase_id == 'MatBase':
				continue
			phase = sp_dummy.dict_phases[phase_id]
			sp_dummy.dict_phases = xrdanal.phase( sp_dummy.dict_phases, phase[6], phase_id, phase[0], phase[1], maj = True )

		sp_dummy.rietvelt_calc()
		f_count = interp1d( sp_dummy.fit.data_reg.theta, sp_dummy.fit.data_reg.count, fill_value=0. )
		
		
		return f_count( theta )
	
	#Résolution du système
	sigma = self.raw_data.count**0.5
	x, Vx = curve_fit( f, self.raw_data.theta, self.raw_data.count, p0 = p0, sigma = sigma, bounds = ( bound_min, bound_max ), verbose = 2 )

	#Transfert de la solution dans la structure 'spectre'
	for i in range( len( x ) ):
		arg = list_args[i]
		try:
			arg1, arg2, arg3 = arg.split(':')
		except:
			arg1 = ''
			arg2 = ''
			arg3 = ''

		if arg[0] == 'b':
			ptno = int( arg[-1] )
			self.back_pts_list[ptno][1] = x[i]
			self.back_pts_list[ptno][2] = Vx[i, i]**0.5
		elif arg1 == 'p':
			phase_id = arg2
			phase = self.dict_phases[phase_id]
			if arg3 == 'x':
				phase[7] = (x[i], Vx[i, i]**0.5)
			elif arg3 == 'a':
				phase[2] = (x[i], Vx[i, i]**0.5)
			elif arg3 == 'c':
				phase[3] = (x[i], Vx[i, i]**0.5)
			elif arg3 == 's':
				phase[8] = (x[i], Vx[i, i]**0.5)
			elif arg3[0] == 'C':
				phase[9][arg3] = (x[i], Vx[i, i]**0.5)
		elif arg == 'i:ra:ra':
			self.instrum_data.ratio_alpha = x[i]
		elif arg[0:5] == 'i:dth':
			if arg[-1] == 'c':
				self.instrum_data.corr_th[0] = (x[i], Vx[i, i]**0.5)
			elif arg[-1] == 's':
				self.instrum_data.corr_th[1] = (x[i], Vx[i, i]**0.5)
			elif arg[-1] == '1':
				self.instrum_data.corr_th[2] = (x[i], Vx[i, i]**0.5)
			elif arg[-1] == '2':
				self.instrum_data.corr_th[3] = (x[i], Vx[i, i]**0.5)
		elif arg[0:6] == 'i:FWHM':
			if arg[-1] == 'U':
				self.instrum_data.calib_FWHM[0] = (x[i], Vx[i, i]**0.5)
			elif arg[-1] == 'V':
				self.instrum_data.calib_FWHM[1] = (x[i], Vx[i, i]**0.5)
			elif arg[-1] == 'W':
				self.instrum_data.calib_FWHM[2] = (x[i], Vx[i, i]**0.5)
		elif arg[0:4] == 'i:ff':
			if arg[-1] == 'b':
				self.instrum_data.calib_ff[0] = (x[i], Vx[i, i]**0.5)
			elif arg[-1] == 'c':
				self.instrum_data.calib_ff[1] = (x[i], Vx[i, i]**0.5)
	
	FV_tot = 0.
	var_tot = 0.
	for arg in self.dict_args:
		if self.dict_args[arg][1] == 'B':
			phasemax = arg[2:-2]

	for phase_id in self.dict_phases:
		if phase_id == 'MatBase' or phase_id == phasemax:
			continue
		FV_tot += self.dict_phases[phase_id][7][0]
		var_tot += self.dict_phases[phase_id][7][1]**2


	self.dict_phases[ phasemax ][7] = ( 1 - FV_tot, var_tot**0.5 )
	
	for phase_id in self.dict_phases:
		if phase_id == 'MatBase':
			continue
		phase = self.dict_phases[phase_id]
		self.dict_phases = xrdanal.phase( self.dict_phases, phase[6], phase_id, phase[0], phase[1], maj = True )

	self.fit.list_args = list_args
	self.fit.x = x
	self.fit.Vx = Vx
	self.etat.rietvelt_opt = True
	self.rietvelt_calc( affich = affich )


def rietvelt_params( self ):
	print( 'Argument  Valeur      Sigma      Actif  Min         Max' )
	for arg in self.dict_args:
		try:
			arg1, arg2, arg3 = arg.split(':')
		except:
			arg1 = ''
			arg2 = ''
			arg3 = ''

		if arg[0] == 'b':
			ptno = int( arg[1:] )
			x, sig_x =  self.back_pts_list[ptno][1], self.back_pts_list[ptno][2]
		elif arg1 == 'p':
			phase_id = arg2
			phase = self.dict_phases[phase_id]
			if arg3 == 'x':
				x, sig_x = phase[7]
			elif arg3 == 'a':
				x, sig_x = phase[2]
			elif arg3 == 'c':
				x, sig_x = phase[3]
			elif arg3 == 's':
				x, sig_x = phase[8]
			elif arg3[0] == 'C':
				x, sig_x = phase[9][arg3]
		elif arg1 == 'i' and arg2 == 'ra':
			x = self.instrum_data.ratio_alpha
			sig_x = 0.
		elif arg1 == 'i' and arg2 == 'dth':
			if arg3 == 'c':
				x, sig_x = self.instrum_data.corr_th[0]
			elif arg3 == 's':
				x, sig_x = self.instrum_data.corr_th[1]
			elif arg3 == '1':
				x, sig_x = self.instrum_data.corr_th[2]
			elif arg3 == '2':
				x, sig_x = self.instrum_data.corr_th[3]
		elif arg1 == 'i' and arg2 == 'FWHM':
			if arg3 == 'U':
				x, sig_x = self.instrum_data.calib_FWHM[0]
			elif arg3 == 'V':
				x, sig_x = self.instrum_data.calib_FWHM[1]
			elif arg3 == 'W':
				x, sig_x = self.instrum_data.calib_FWHM[2]
		elif arg1 == 'i' and arg2 == 'ff':
			if arg[-1] == 'b':
				x, sig_x = self.instrum_data.calib_ff[0]
			elif arg[-1] == 'c':
				x, sig_x = self.instrum_data.calib_ff[1]

		if self.dict_args[arg][1] == 'B':
			print( '%8s %11.4e %11.4e %5r %11.4e %11.4e' %( arg, x, sig_x, self.dict_args[arg][0], 0., 1. ))
		else:
			print( '%8s %11.4e %11.4e %5r %11.4e %11.4e' %( arg, x, sig_x, self.dict_args[arg][0], self.dict_args[arg][1][0], self.dict_args[arg][1][1] ) ) 
	
	print( '\n***\n' )
	print( u'Écart-type\t\tStandard\t\tCorrigé [Berar1991]' )
	for phase_id in self.dict_phases:
		if phase_id == 'MatBase':
			continue

		print( 'Phase : ' + phase_id )
		print( 'Fraction volumique    :\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][7] ) + '\t\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][7][0], self.fit.fact_corr**0.5*self.dict_phases[phase_id][7][1] ) )
		print( u'Paramètre de maille a :\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][2] ) + '\t\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][2][0], self.fit.fact_corr**0.5*self.dict_phases[phase_id][2][1] ) )
		if self.dict_phases[phase_id][1] == 'BCT' or self.dict_phases[phase_id][1] == 'HCP':
			print( u'Paramètre de maille c :\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][3] ) + '\t\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][3][0], self.fit.fact_corr**0.5*self.dict_phases[phase_id][3][1] ) )
		print( u'Déformation s         :\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][8] ) + '\t\t' + xrdanal.write_nmbr( self.dict_phases[phase_id][8][0], self.fit.fact_corr**0.5*self.dict_phases[phase_id][8][1] ) + '\n')


def correl_params( spectre, correl_min = 0.75 ):
	Vx = spectre.fit.Vx
	diagmat = np.diagflat( np.diagonal( Vx )**-0.5 )
	corrmat = np.matmul( diagmat, np.matmul( Vx, diagmat) )
	print( 'R > %f : ' %correl_min )
	for i in range( len( corrmat ) ):
		for j in range( len( corrmat ) ):
			if np.abs( corrmat[i, j] ) > correl_min:
				print( '( %d, %d ) : %f ' %(i, j, corrmat[i, j]) )


#def f_h( phi, th_obs, L ):
#	try:
#		return L*((math.cos(phi*2*np.pi/360)/math.cos(th_obs*2*np.pi/360))**2 - 1.)**0.5
#
#	except ValueError:
#		return 0.
#
#	except TypeError:
#		n = len(phi)
#		phi_vec = np.zeros(n)
#		for i in range(n):
#			phi_vec[i] = f_h(phi[i])
#
#		return phi_vec	


#def f_W( phi, th_obs, phi_min, phi_infl, H, S, L ):
#	try:
#		if phi < phi_min:
#			return 0.
#		elif phi > th_obs:
#			return 0.
#		elif phi < phi_infl:
#			return H + S - f_h(phi, th_obs, L)
#		elif phi < th_obs:
#			return 2*S
#		else:
#			return 0.
#
#	except TypeError:	
#		n = len(phi)
#		phi_vec = np.zeros(n)
#		for i in range(n):
#			phi_vec[i] = f_W(phi[i])
#
#		return phi_vec	

#def f_D( phi, th_obs, phi_min, phi_infl, H, S, L ):
#	try:
#		return L*f_W(phi, th_obs, phi_min, phi_infl, H, S, L)/(2*H*f_h(phi, th_obs, L)*math.cos(phi*2*np.pi/360))
#
#	except ZeroDivisionError:
#		return 0.
#
#	except TypeError:
#		n = len(phi)
#		phi_vec = np.zeros(n)
#		for i in range(n):
#			phi_vec[i] = f_D(phi[i])
#
#		return phi_vec	

#def intD( th_obs, phi_min, phi_infl, H, S, L ):
#	th0 = th_obs*2*np.pi/360
#	a = 0.5*(th0 + np.pi/2)
#	#Intégrale contenant fonction W pour th0>thinfl, évaluée avec la singularité par la méthode des résidus
#	f_res = lambda t: 1j*a*np.exp(1j*t)/(np.cos(a*np.exp(1j*t))*( (np.cos(a*np.exp(1j*t)))**2/(np.cos(th0))**2 - 1)**0.5 )
#	I_res = -np.min([H, S])/H*quad(f_res, 0, np.pi/2)[0].real
#
#	#On doit soustraire à l'intégrale ci-dessus pour conserver seulement la plage entre l'inflexion et le max
#	f_soustr = lambda t: 1./(np.cos(t)*( (np.cos(t))**2/(np.cos(th0))**2 - 1)**0.5)
#	I_soustr = np.min([H, S])/H*quad( f_soustr, 0, phi_infl*2*np.pi/360 )[0]
#
#	#On rajoute ensuite le reste de la place, soit phi_min<phi<phi_infl
#	f_min_infl = lambda t: f_W(t*360./2/np.pi, th_obs, phi_min, phi_infl, H, S, L )/(2*H*np.cos(t)*( (np.cos(t))**2/(np.cos(th0))**2 - 1)**0.5)
#	I_min_infl = quad(f_min_infl, phi_min*2*np.pi/360, phi_infl*2*np.pi/360)[0]
#
#	return (I_res - I_soustr + I_min_infl)*360/2/np.pi

