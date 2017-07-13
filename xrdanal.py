# -*- coding: latin-1 -*-

#######################################################################
# Projet            : DPHM601 Fragilisation par l'hydrogène 
#
# Programme         : xrdanal.py
#
# Auteur            : Simon Laliberté-Riverin
#
# Date de création  : 20160206
#
# Objectif          : Quantification des phases à l'aide d'une régression de diffraction des rayons X obtenue avec xrdsim 
#
# Historique rév.   :
#
# Rév. 	Date    	Auteur      	Description  
#
# 0	20170206	S. L.-Riverin	Transféré les fonctions read_el, read_melange et mat_conv de xrdsim. Créé calc_surf.
# 1	20170313	S. L.-Riverin	Créé les fonctions phase, calc_d, scatt_f, pol_f, lor_f, temp_f et phiofx.
# 2	20170320	S. L.-Riverin	Correction erreur dans fonction quant et ajout correction pour dispersion anormale dans phase.
# 3	20170327	S. L.-Riverin	Rajouté calcul paramètre de maille moyen
#
#			CHANGEMENTS SUBSÉQUENTS DANS GITHUB
#
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.special import wofz, erfc
from scipy.optimize import fsolve
from scipy.integrate import simps
import os.path
import csv
this_dir, this_filename = os.path.split( __file__ )
DATA_PATH = os.path.join( this_dir, "data" )
import xrdsim

def calc_surf( spectre ):
	if spectre.etat.calcul == False:
		print( u'Compléter la simulation d\'abord' )
		return

	print( u'Pic #\tPhase\tPlan\t\t2-theta\tI\tA\tFWHM\tForm Fact  P/B\t    P/N' )
		
	surf_phase = {}
	surf_tot_PSF = 0

	for i in range( len( spectre.peak_list ) ):
		pic_index = spectre.peak_list[i][0]
		if len( spectre.peak_list[i][1] ) == 0:
			continue	
		PSF = spectre.peak_list[i][1][0]
		phase = spectre.peak_list[i][4]
		plan = spectre.peak_list[i][5]
		
		if PSF == 'v':
			I0g = spectre.peak_list[i][1][1]
			I0l = spectre.peak_list[i][1][2]
			th0 = spectre.peak_list[i][1][3]
			beta_g = spectre.peak_list[i][1][4]
			beta_l = spectre.peak_list[i][1][5]
			k_voigt = beta_l / (np.pi**0.5 * beta_g)
			A = beta_g*beta_l*I0g*I0l
			f_w = lambda w: wofz(np.pi**0.5*w/beta_g + 1j*k_voigt).real - 0.5*wofz(1j*k_voigt).real
			w = abs(fsolve( f_w, 0.5 ))
			FWHM = 2*w
			I = beta_l*I0g*I0l*wofz(1j*k_voigt).real
			beta = beta_g * np.exp(-k_voigt**2)/(erfc(k_voigt))
			f_fact = FWHM / beta

		elif PSF == 'v2' or PSF[0:3] == 'v2k' or PSF[0:3] == 'v3k':
			I0lg = spectre.peak_list[i][1][1]
			th0 = spectre.peak_list[i][1][2]
			beta_g = spectre.peak_list[i][1][3]
			beta_l = spectre.peak_list[i][1][4]
			k_voigt = beta_l / (np.pi**0.5 * beta_g)
			A = beta_g*beta_l*I0lg**2
			f_w = lambda w: wofz(np.pi**0.5*w/beta_g + 1j*k_voigt).real - 0.5*wofz(1j*k_voigt).real
			w = abs(fsolve( f_w, 0.5 ))
			FWHM = 2*w
			I = beta_l*I0lg**2*wofz(1j*k_voigt).real
			beta = beta_g * np.exp(-k_voigt**2)/(erfc(k_voigt))
			f_fact = FWHM / beta

		try:
			surf_phase[phase] += A
		except KeyError:
			surf_phase[phase] = A

		surf_tot_PSF += A
		
		PB = 1 + I / np.mean(spectre.data_back.count)
		PN = 1 + I / spectre.fit.noise
		print( str(pic_index) + '\t' + phase + '\t' + str(plan) + '\t' + str(round(th0,3)) + '\t' + str(round(I,1)) + '\t' + str(round(A,2)) + '\t' + str(round(FWHM,3)) + '\t' + str(round(f_fact,5)) + '\t   ' + "%-5.3g" % PB + '\t' + "%-5.3g" % PN )
	
	surf_tot_back = simps( spectre.data_back.count, spectre.data_back.theta )
	
	print( '---\nPhase\t\tA_tot\t\t% tot phases\t% tot' )
	for key, value in surf_phase.iteritems():
		print('\t' + key + '\t' + str(round(value,0)) + '\t\t' + str(100*round(value/surf_tot_PSF, 3)) + '\t\t' + str(100*round(value/(surf_tot_PSF+surf_tot_back),3)) )

	print( '---\nTotal pics\t' + str(round(surf_tot_PSF, 0))+ '\t\t100\t\t' + str(100*round(surf_tot_PSF/(surf_tot_PSF+surf_tot_back),3)))
	print( u'Arrière-plan\t' + str(round(surf_tot_back, 0)) + '\t\t\t\t' + str(100*round(surf_tot_back/(surf_tot_PSF+surf_tot_back),3)))
	print( '---\nTotal\t\t' + str(round(surf_tot_PSF+surf_tot_back,0)) + '\t\t\t100' )
	print( u'---\nDonnées brutes\t' + str(round(simps( spectre.raw_data.count, spectre.raw_data.theta )))) 

	print( '---\nBruit\t\t' + str(round(spectre.fit.noise, 0 ) ) )
	print( 'Fond moyen\t' + str(round(np.mean(spectre.data_back.count) ) ) )


def phase( phase_list, mat, phase_id, descr, cryst_struct, emetteur = 'Cu', raie = 'a', a = 0, c = 0, pourc_C = 0, alpha = 13.3*2*np.pi/360, anom_scatt = True, affich = 0, liste_pics = [], liste_p = [] ):
	"""
	Rajoute ou modifie une phase dans la liste des phases disponibles pour l'analyse.
	Modifie la variable phase_list.

	Args :
		phase_list : Liste des phases disponibles pour l'analyse
		mat : Composition chimique du matériau  
		phase_id : Identifiant de la phase, qui sera comparé aux "tags" de le la simulation
		desc : Texte descriptif de la phase
		cryst_struct : Structure cristallographique de la phase ('BCC', 'FCC' ou 'HCP')
		emetteur, raie : Source de rayons X. Valeur par défaut : 'Cu' et 'a' pour Cu K-alpha
		a, c : Paramètres de maille (Angstrom)
		pourc_C : Pourcentage de carbone de la microstructure.
		alpha : angle de diffraction du monochromateur. Valeur par défaut 2*alpha = 26.6°, soit l'angle de diffraction
			pour le rayonnement Cu K-alpha d'un monochromateur courbé en graphite pyrolitique, ayant un paramètre d
			de maille de 3.35 Angstroms.
		anom_scatt : Application ou non de la correction pour dispersion "anormale".	
		affich : Mettre à 1 pour imprimer les résultats à l'écran
		liste_pics : modifie la liste de plans par défaut selon la structure. 
			Mettre de la forme liste_pics = [ [(h1, k1, l1)], [(h2, k2, l2)] ... ]
			La liste des facteurs de multiplicité liste_p doit correspondre à liste_pics
		liste_p : liste des facteurs de multiplicité à spécifié en même temps que liste_pics
		
	Autres variables :
		V = Volume d'une maille (Angstrom^3)
		(h, k, l) = Plan de diffraction
		p = Facteur de multiplicité du plan (h, k, l)
		d = Distance interplanaire du plan (h, k, l)
		theta = Position théorique du pic (h, k, l)
		sf = Facteur de dispersion
		f1, f2 = Correction réelle et imaginaire pour dispersion anormale
		FF = Facteur de structure multiplié par son complexe conjugué
		Lf = Facteur de Lorentz
		pf = Facteur de polarisation
		LP = Lf * pf
		Tf = Facteur de température
		R = Facteur global d'échelle de la réflexion (h, k, l)

	Formats :	
		materiau = list
				Exemple de format de liste. 'B' signifie la balance
				[ [ 'Fe-0.4C'   , 'm'		],
				  [ 'C'		, 0.004		],
				  [ 'Fe'	, 'B'		] ]

		phase_list = dict
			{'MatBase', [mat, emetteur, raie]}
			{phase_id, [descr, cryst_struct, a, c, V, liste_pics]}

		liste_pics = list
			[liste_pics[0], liste_pics[1] ... ]

		liste_pics[i] = [(h, k, l), p, d, theta, FF, Lf, pf, Tf, R]

	"""
	
	#Vérifie si cette phase existe déjà.
	if 'MatBase' in phase_list:
		if phase_list['MatBase'][0][0][0] != mat[0][0] or phase_list['MatBase'][1] != emetteur or phase_list['MatBase'][2] != raie:
			print(u'Veuillez vous assurer que le matériau et la longueur d\'onde du rayon X incident sont les mêmes pour toutes les phases')
			return

	else:	
		phase_list['MatBase'] = [mat, emetteur, raie]
	
	if phase_id in phase_list:
		ow = raw_input( u'Phase déjà existante... écraser les données? 1 = oui, autre = non ... : '.encode('cp850') )
		if ow != '1':
			return


	[mu_m, A, rho, lam, f1, f2] = read_melange( mat, emetteur, raie, affich )

	if anom_scatt == True: #Facteurs de correction pour dispersion "anormale". Ici pour Cu K-alpha dans Fe [ITC vol. C table 4.2.6.8]
		if affich == 1:
			print( u'\nCorrection pour dispersion anormale :' )
			print( u'\tRéel : \t\tf1 = ' + str(f1) )
			print( u'\tImaginaire :	f2 = ' + str(f2) )

	else:
		f1 = 0
		f2 = 0

	if cryst_struct == 'BCC':	#Cubique centré
		if a == 0 :
			a = 2.872 #Par défaut, on attribue la valeur usuelle du paramètre de maille du fer alpha
		
		c = 0;
		V = a**3
		
		if liste_pics == []:
			liste_pics = [ [(1, 1, 0)], [(2, 0, 0)], [(2, 1, 1)], [(2, 2, 0)] ]
		if liste_p == []:
			liste_p = [12, 6, 24, 12]

		for i in range(len(liste_pics)):
			d = calc_d( liste_pics[i][0], a )
			theta = np.arcsin( lam / (2*d) )
			s = np.sin(theta) / lam
			sF = scatt_f( s ) + f1 + 1j*f2
			FF = (4*sF*np.conj(sF)).real
			
			p = liste_p[i]
			Lf = lor_f( theta )
			pf = pol_f( theta, alpha )
			Tf = temp_f( s )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			liste_pics[i].append(p)
			liste_pics[i].append(d)
			liste_pics[i].append(theta)
			liste_pics[i].append(FF)
			liste_pics[i].append(Lf)
			liste_pics[i].append(pf)
			liste_pics[i].append(Tf)
			liste_pics[i].append(R)
		
	elif cryst_struct == 'FCC': 	#Cubique face centrée
		if a == 0:
			a = 3.555 + 0.044*pourc_C
		c = 0;
		V = a**3

		if liste_pics == []:
			liste_pics =  [ [(1, 1, 1)], [(2, 0, 0)], [(2, 2, 0)], [(3, 1, 1)], [(2, 2, 2)], [(3, 3, 1)], [(4, 2, 2)] ]
		if liste_p == []:
			liste_p = [8, 6, 12, 24, 8, 24, 24]

		for i in range(len(liste_pics)):
			d = calc_d( liste_pics[i][0], a )
			theta = np.arcsin( lam / (2*d) )
			s = np.sin(theta) / lam
			sF = scatt_f( s, mat ) + f1 + 1j*f2
			FF = (16*sF*np.conj(sF)).real
			p = liste_p[i]
			Lf = lor_f( theta )
			pf = pol_f( theta, alpha )
			Tf = temp_f( s )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			liste_pics[i].append(p)
			liste_pics[i].append(d)
			liste_pics[i].append(theta)
			liste_pics[i].append(FF)
			liste_pics[i].append(Lf)
			liste_pics[i].append(pf)
			liste_pics[i].append(Tf)
			liste_pics[i].append(R)

	elif cryst_struct == 'DC':	#Diamond Cubic
		if a == 0:
			a = 5.4309	#Silicium selon CIF 27-1402
		c = 0;
		V = a**3

		if liste_pics == []:
			liste_pics =  [ [(1, 1, 1)], [(2, 2, 0)], [(3, 1, 1)], [(4, 0, 0)], [(3, 3, 1)], 
					[(4, 2, 2)], [(5, 1, 1)], [(4, 4, 0)], [(5, 3, 1)], [(6, 2, 0)],
					[(5, 3, 3)] ]
		if liste_p == []:
			liste_p = [	8, 12, 24, 6, 24,
					24, 24, 12, 48, 24,
					24 ]

		for i in range(len(liste_pics)):
			(h, k, l) = liste_pics[i][0]
			d = calc_d( liste_pics[i][0], a )
			theta = np.arcsin( lam / (2*d) )
			s = 1./(2*d)
			sF = scatt_f( s, mat ) + f1 + 1j*f2
			if (h + k + l) % 2 == 0:	#Somme des indices paire
				FFmult = 64
			else:
				FFmult = 32
			FF = FFmult*(sF*np.conj(sF)).real
			p = liste_p[i]
			Lf = lor_f( theta )
			pf = pol_f( theta, alpha )
			Tf = temp_f( s )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			liste_pics[i].append(p)
			liste_pics[i].append(d)
			liste_pics[i].append(theta)
			liste_pics[i].append(FF)
			liste_pics[i].append(Lf)
			liste_pics[i].append(pf)
			liste_pics[i].append(Tf)
			liste_pics[i].append(R)

	elif cryst_struct == 'HCP':	#Hexagonal compact
		V = 3**0.5*a**2*c/2

		if liste_pics == []:
			liste_pics = [  [(0, 0, 2)], [(1, 0, 0)], [(1, 0, 1)], [(1, 0, 2)], [(1, 0, 3)], [(1, 1, 0)], 
					[(0, 0, 4)], [(1, 1, 2)], [(2, 0, 0)], [(2, 0, 1)], [(1, 0, 4)], [(2, 0, 2)],
					[(2, 0, 3)], [(1, 0, 5)], [(1, 1, 4)], [(2, 1, 0)], [(2, 1, 1)], [(2, 0, 4)],
					[(0, 0, 6)], [(2, 1, 2)], [(1, 0, 6)], [(2, 1, 3)], [(3, 0, 0)], [(2, 0, 5)],
					[(3, 0, 2)], [(2, 1, 4)] ]

		if liste_p == []:	
			liste_p = [	2, 6, 12, 12, 12, 6,
					2, 12, 6, 12, 12, 12,
					12, 12, 12, 12, 24, 12,
					2, 24, 12, 24, 6, 12,
					12, 24 			] 

		for i in range( len(liste_pics) ):
			(h, k, l) = liste_pics[i][0]
			d = calc_d( (h, k, l), a, c, 'h' )
			theta = np.arcsin( lam / (2*d) )
			s = np.sin(theta) / lam
			sF = scatt_f( s ) + f1 + 1j*f2
			FF = 4 * (sF*np.conj(sF)).real * (np.cos( np.pi*( (h+2*k)/3.+l/2.) ) )**2
			p = liste_p[i]
			Lf = lor_f( theta )
			pf = pol_f( theta, alpha )
			Tf = temp_f( s )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			liste_pics[i].append(p)
			liste_pics[i].append(d)
			liste_pics[i].append(theta)
			liste_pics[i].append(FF)
			liste_pics[i].append(Lf)
			liste_pics[i].append(pf)
			liste_pics[i].append(Tf)
			liste_pics[i].append(R)



	phase_list[phase_id] = [descr, cryst_struct, a, c, V, liste_pics]
	return phase_list

def plot_phase( phase, dict_phases, raw_data = [], couleur = 'blue', theta_min = 20, theta_max = 120 ):
	"""

	Trace les pics de une ou plusieurs phases dans une liste de phase
	Peut comparer ces pics avec les données brutes du spectre.

	Args :
		phase : Identificateur de phase, ou liste d'identificateurs de phase (phase_id) à retrouver dans la liste des phases
		dict_phases : Liste des phases
		raw_data (optionnel) : Structure xrdsim.data contenant les données brutes à tracer
		couleur (optionnel) : couleur des pics à tracer pour la phase 'phase'. Utilisé principalement lors de la récursion
		theta_min, theta_max : Minimum et maximum d'affichage des pics. Si des données brutes sont données, ces valeurs seront 
			écrasées par les limites du spectre.

	"""

	plt.ion()

	if raw_data != [] :
		plt.plot(raw_data.theta, raw_data.count/max(raw_data.count), color = 'black', label = r'Donn\'{e}s brutes' )
		theta_min = min(raw_data.theta)
		theta_max = max(raw_data.theta)

	plt.axis([theta_min, theta_max, 0, 1.1])

	if type( phase ) == list:
		cm = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] 
		for i in range( len( phase ) ):
			plot_phase( phase[i], dict_phases, couleur = cm[i], theta_min = theta_min, theta_max = theta_max )

		return	

	phase_data = dict_phases[ phase ]

	R_max = 0
	for i in range( len( phase_data[5] ) ):
		R = phase_data[5][i][8]
		if R > R_max:
			R_max = R
	
	for i in range( len( phase_data[5] ) ):
		angle_2theta = 2*phase_data[5][i][3] * 360. / (2.*np.pi)
		if raw_data != [] and ( angle_2theta > theta_max or angle_2theta < theta_min ):
			continue
		R = phase_data[5][i][8] / R_max
		plt.plot([angle_2theta, angle_2theta], [0, R], color = couleur)
		plt.annotate( str( phase_data[5][i][0] ), [angle_2theta, 0.5 * R] )

	plt.plot( [], [], color = couleur, label = phase )
	plt.legend()



def calc_d( hkl, a, c = 0, cryst_syst = 'c' ):
	"""

	Calcule la distance interplanaire du plan (h, k, l), pour une structure cubique ayant un paramètre de maille a.

	Args :
		hkl : 	Tuple contenant les indices du plan : hkl = (h, k, l)
		a, c :	Paramètres de maille (par défaut c = 0)
		cryst_syst : Système cristallographique :
			'c' : Cubique (par défaut)
			'h' : Hexagonal

	"""	
	
	h = hkl[0]
	k = hkl[1]
	l = hkl[2]
	
	if cryst_syst == 'c':
		return np.sqrt( a**2 / (h**2 + k**2 + l**2) )
	elif cryst_syst == 'h':
		return ( 4./3.*(h**2 + h*k + k**2)/a**2 + l**2/c**2 )**(-0.5)

def calc_a( hkl, theta, lam ):
	"""

	Estime le paramètre de maille en fonction de l'angle observé sur la réflexion (hkl)

	"""

	h = hkl[0]
	k = hkl[1]
	l = hkl[2]
	
	d = lam / (2*np.sin(theta))
	return d * np.sqrt( h**2 + k**2 + l**2 )
		
def scatt_f( s, mat = 'Fe' ):
	"""

	Calcul du facteur de diffusion (scattering factor) atomique.
	Utilise l'interpolation f(s) = sum_1,4 (ai*exp( -bi*s**2 ) + c
	Réf. : International Tables for Crystallography, vol. C section 6.1
	
	où :
		s : 	 sin( theta ) / lambda
		theta :  angle de diffraction
		lambda : longueur d'onde du rayon X incident
	
	Args :
		mat : Matériau calculé (Fe par défaut) -> Accepte une liste de type 'Matériau'
		s : Tel que défini plus haut

	"""

	#Si l'utilisateur fournit une liste d'angle, une récursion est faite pour chacun des angles et un vecteur est retourné
	if type(s) == np.ndarray or type(s) == list: 
		s_f = np.array([])

		for j in range( len(s) ):
			s_f_temp = scatt_f( s[j], elem )
			s_f = np.append( s_f, s_f_temp )

	#Ici, un seul angle est étudié
	elif type(s) == int or type(s) == float or type(s) == np.float64:

		#S'il s'agit d'un mélange de plusieurs éléments, une récursion est faite pour trouver le facteur moyen pondéré
		#selon la fraction atomique
		if type( mat ) == list:
			mat_conv( mat, 'a' )
			s_f_vect = []
			for i in range( len(mat) - 1):
				mat_temp = mat[i+1][0]
				n = mat[i+1][1]
				s_f_vect.append( scatt_f( s, mat_temp ) * n )
			
			s_f = np.sum( s_f_vect )

		#Un seul élément, substance pure
		elif type( mat ) == str:
			X = read_el( mat )
			a = X[6]
			b = X[7]
			c = X[8]

			s_f = c
			for i in range(4):
				s_f += a[i]*np.exp( -b[i]*s**2 )

			if s > 2: 
				print( u'Attention : s > 2, corrélation imprécise' )
				
	return s_f


def pol_f( theta, alpha = 13.3*2*np.pi/360 ):
	"""
	Fonction calculant le facteur de polarisation [ITC vol. C., sec. 6.2.2]

	Sans monochromateur : pf = 0.5*(1 + cos^2(2*theta))
	Avec monochromateur : pf = (1 + cos^2(2*alpha)*cos^2(2*theta) )/(1 + cos^2(2*alpha)

	Args :
		theta = angle de diffraction de l'échantillon. Accepté sous forme de scalaire ou de vecteur.
		alpha = angle de diffraction du monochromateur. Valeur par défaut 2*alpha = 26.6°, soit l'angle de diffraction
			pour le rayonnement Cu K-alpha d'un monochromateur courbé en graphite pyrolitique, ayant un paramètre d
			de maille de 3.35 Angstroms.
			

	Retourne pf, le facteur de polarisation, sous forme scalaire ou vectorielle.

	"""

	return (1. + np.cos(2.*alpha)**2. * np.cos(2.*theta)**2. ) /  (1. + np.cos(2.*alpha)**2. )


def lor_f( theta ):
	"""
	Fonction calculant le facteur de Lorentz [ITC vol. C, sec. 6.2.5]

	Lf = 1 / ( sin^2(theta)*cos(theta) )

	Arg :
		theta = angle de diffraction de l'échantillon

	"""

	return 1 / (np.sin(theta)**2 * np.cos(theta))


def temp_f( s, mat = 'Fe', T = 293. ):
	"""
	Fonction calculant le facteur de la température exp(-2*M)  [Cullity]

	où :
		M = (6*h**2*T)/(m*k*DebyeT**2) * [phiofx(x) + x/4]*s**2
		... = 1.1490292193e4 * T/(A*DebyeT) * [phiofx(x) + x/4]*s**2
		h = Constante de Planck = 6.626e.34 m^2.kg/s
		m = Masse d'un atome = A / N
		A = Masse atomique (Valeur par défaut : Fer )
		N = Nombre d'Avogadro = 6.022e23 mol^(-1)
		k = Constante de Boltzmann = 1.381e-23 m^2.kg/(s^2.K)
	(arg)	T = Température (Valeur par défaut 293 K : Température ambiante)	
		DebyeT = Température de Debye (Valeur par défaut : Fer)
		x = DebyeT / T
		phiofx(x) = Fonction de Debye = 1/x * int_0^(inf) t*dt/(exp(t)-1)
	(arg)	s = sin(theta)/lambda ( Angstrom^(-1) )
		theta = Angle de diffraction (degrés)
		lambda = Longueur d'onde des rayons X incidents (Angstrom)
	(arg)	mat = Matériau, sous forme d'élément (str) ou mélange (list)
	

	"""

	if type( mat ) == str:
		X = read_el( mat )
		DebyeT = X[5]
		A = X[2]
		x = DebyeT / T
		M = 1.1490292193e4 * T / (A*DebyeT**2) * (phiofx(x) + x/4)*s**2
		return np.exp(-2*M)

	elif type( mat ) == list:
		mat_conv( mat, 'a' )
		T_f_vect = []
		for i in range( len(mat) - 1):
			mat_temp = mat[i+1][0]
			n = mat[i+1][1]
			T_f_vect.append( temp_f( s, mat_temp, T ) * n )
		
		return np.sum( T_f_vect )

def phiofx(x):
	"""
	Fonction calculant numériquement la valeur de la fonction de Debye.
	
	phiofx(x) = int_0^(inf) t*dt/(exp(t)-1)
	
	Pour 0 < x < 0.1 : 	On fait une expansion en séries de Taylor de l'intégrande, qu'on peut ensuite intégrer facilement.
				Les cinq premiers termes suffisent pour obtenir une précision suffisante.
				Merci à Mark Fischler qui m'a proposé cette stratégie sur StackExhchange

	Pour x > 0.1 :		On calcule premièrement la valeur de l'intégrale à x = 0.1 de la façon montrée plus haut.
				On calcule ensuite la valeur de l'intégrale entre 0.1 et x avec la méthode de Simpson.

	La fonction accepte l'argument x en scalaire ou en vecteur.			

	"""

	
	phi_taylor = lambda x : 1. - x/4. + x**2/36. -x**4/180. + x**6/211680.
	integrand = lambda x : x/(np.exp(x) - 1.)

	if type(x) == list or type(x) == np.ndarray:
		phi = np.array([])
		for i in range(len(x)):
			phi = np.append(phi,  phiofx(x[i]) )
		
		return phi			

	else:
		if x <= 0.1:
			return phi_taylor(x)
		else:
			x_range = np.arange(0.1, x, 0.000001)
			integrand_range = integrand(x_range)
			return 1./x * simps(integrand_range, x_range) + 0.1/x*phi_taylor(0.1)





def read_el( absorbeur, emetteur = 'Cu', raie = 'a', affich = 0 ):
	"""

	Retourne les informations pour l'analyse de DRX d'un élément

	Intrants:
		absorbeur : 	chaine contenant le symbole chimique de l'absorbeur (ex. 'Fe')
		emetteur :	chaine contenant le symbole chimique de l'emetteur (ex. 'Cu')
		raie :		'a' = k-alpha; 'b' = k-beta
		affich :	Imprimer les résultats a l'ecran (Bool.)

	Extrants:
		[mu_m, Z, A, rho, lambda, DebyeT, a, b, c, f1, f2]
		mu_m : 		Coefficient d'absorption massique (cm2/g)
		Z : 		Numéro atomique
		A :		Masse atomique (g/mol)
		rho :		Densité (g/cm3)
		lam :		Longueur d'onde du rayonnement incident (Angstrom)
		DebyeT :	Température de Debye (K)
		a, b, c :	Coefficients pour le calcul du facteur de diffusion
			a = [a1, a2, a3, a4]
			b = [b1, b2, b3, b4]
		f1, f2 :	Facteurs réel et imaginaire de correction du facteur de diffusion	

	"""
	#Choisit la bonne colonne pour la lecture des informations
	if emetteur == 'Mo':
		rowno = 4
	elif emetteur == 'Cu':
		rowno = 6
		rowno2 = 22
	elif emetteur == 'Co':
		rowno = 8
		rowno2 = 24
	elif emetteur == 'Cr':
		rowno = 10
	else:
		print( u'Erreur, émetteur inexistant' )
		return

	if raie == 'b':
		rowno += 1
		raie = 'Beta'
	elif raie == 'a':
		raie = 'Alpha'
	else:
		print( 'Erreur, raie inexistante' )
		return


	if not os.path.exists( os.path.join( DATA_PATH, 'Coeff_abs.csv' ) ):
		print( u'Erreur, fichier de données introuvable' )
		print DATA_PATH
		return

	with open( os.path.join( DATA_PATH, 'Coeff_abs.csv' ), 'rb' ) as csvfile:
		csvreader = csv.reader( csvfile, delimiter = ',' )
		for row in csvreader:
			if row[0] == 'Lambda':
				lam = float( row[rowno] )
			elif row[0] == absorbeur:
				Z = int( row[1] )
				A = float( row[2] )
				rho = float( row[3] )
				mu_m = float( row[rowno] )
				DebyeT = float( row[12] )
				a = [ float( row[13] ), float( row[14] ), float( row[15] ), float( row[16] ) ]
				b = [ float( row[17] ), float( row[18] ), float( row[19] ), float( row[20] ) ]
				c = float( row[21] )
				f1 = float( row[rowno2] )
				f2 = float( row[rowno2 + 1] )
				mu_l = mu_m*rho
				if affich == 1 or affich == 2 or affich == 3:
					print( u'\n----\nÉmetteur :\t\t' + emetteur + ' K-' + raie )
					print( 'Longueur d\'onde :\t' + str( lam ) +'\tAngstrom' )
					print( '\n' )		
					print( 'Absorbeur :\t\t' + absorbeur )
					print( u'Numéro atomique :\t' + str( Z ) )
					print( 'Masse atomique :\t' + str( A ) +'\tg/mol' )
					print( u'Densité :\t\t' + str( rho ) + '\tg/cm3' )
					print( 'Coeff. abs. mass. :\t' + str( mu_m ) + '\tcm2/g' )
					print( 'Coeff. abs. lin. :\t' + str( mu_l ) + '\tcm-1')
					print( 'Temp. de Debye :\t' + str( DebyeT ) + '\tK' )
					print( 'Facteurs de correction pour diffusion anormale :' )
					print( '\tf1 :\t\t' + str( f1 ) )
					print( '\tf2 :\t\t' + str( f2 ) )

				if affich == 2 or affich == 3:
					z=np.arange(0, 5/mu_l, 0.05/mu_l)
					I=np.exp( -mu_l*z )
					leg = emetteur + ' K-' + raie + ' dans ' + absorbeur  
					plt.plot( z, I, label = leg)
					plt.xlabel( 'Epaisseur (cm)' )
					plt.ylabel( 'Intensite transmise I/I0' )
					plt.legend( )
					if affich == 2:
						plt.show()
				return [mu_m, Z, A, rho, lam, DebyeT, a, b, c, f1, f2]

	print 'Erreur, absorbeur inexistant'		


def mat_conv( mat, type_fract, affich = 0 ):
	
	"""
	Convertit la composition chimique d'un matériau de massique vers atomique et vice-versa
	Les caractéristiques du matériau doivent être contenues dans le fichier de donnees
	Intrants :
		mat :		Composition chimique du materiau. Format de type list. Voir ci-dessous
		type_fract :	Type de fraction voulue ('m' = massique, 'a' = atomique)

	Extrants :
		mat_conv :	Composition chimique convertie

	Formats :
		materiau :	Exemple de format de liste. 'B' signifie la balance
				[ [ 'Fe-0.4C'   , 'm'		],
				  [ 'C'		, 0.004		],
				  [ 'Fe'	, 'B'		] ]
	"""

	if type_fract == mat[0][1]:
		if affich == 1:
			print( u'Déjà dans le bon format' )
		tot = 0.
		index_B = 0
		for i in range( len(mat) - 1 ):
			if mat[i + 1][1] == 'B':
				index_B = i + 1
			else:
				tot += mat[i + 1][1]
		if index_B != 0:
			mat[index_B][1] = 1 - tot	
		return mat

	elif type_fract == 'a' and mat[0][1] == 'm':
		tot = 0.
		n_vec = []
		for i in range( len(mat) - 1 ):
			if mat[i + 1][1] == 'B':
				mat = mat_conv( mat, 'm' )
			
			A = read_el( mat[i + 1][0], 'Cu', 'a')[2]
			tot += mat[i + 1][1]
			n_vec.append( mat[i + 1][1] / A )

		for i in range( len( n_vec ) ):
			mat[i + 1][1] = n_vec[i]/sum(n_vec)

		mat[0][1] = 'a'
		return mat

	elif type_fract == 'm' and mat[0][1] == 'a':
		tot = 0.
		n_vec = []
		for i in range( len(mat) - 1 ):
			if mat[i + 1][1] == 'B':
				mat = mat_conv( mat, 'a' )
			
			A = read_el( mat[i + 1][0], 'Cu', 'a')[2]
			tot += mat[i + 1][1]
			n_vec.append( mat[i + 1][1] * A )

		for i in range( len( n_vec ) ):
			mat[i + 1][1] = n_vec[i]/sum(n_vec)

		mat[0][1] = 'm'
		return mat

def read_melange( mat, emetteur = 'Cu', raie = 'a', affich = 0 ):
	"""
	Retourne les informations pour l'analyse de DRX d'un élément
	Intrants:
		mat : 	Composition chimique du matériau (format liste)
		emetteur :	chaine contenant le symbole chimique de l'emetteur (ex. 'Cu')
		raie :		'a' = k-alpha; 'b' = k-beta
		affich :	Imprimer les résultats a l'ecran (Bool.)

	Extrants:
		[mu_m, Z, A, rho, lambda, f1, f2]
		mu_m : 		Coefficient d'absorption massique moyen (cm2/g)
		A :		Masse atomique moyenne (g/mol)
		rho :		Densité moyenne (g/cm3)
		lam :		Longueur d'onde du rayonnement incident (Angstrom)
		f1, f2 :	Facteurs réel et imaginaire de correction du facteur de diffusion	
	
	"""
	mat_conv( mat, 'm' )
	V_vec = []
	n_vec = []
	m_vec = []
	mu_m_vec = []
	f1_vec = []
	f2_vec = []

	for i in range( len(mat) -1 ):
		X = read_el( mat[i + 1][0], emetteur, raie )
		mu_m = X[0]
		Z = X[1]
		A = X[2]
		rho = X[3]
		lam = X[4]
		f1 = X[9]
		f2 = X[10]

		m = mat[i + 1][1]
		m_vec.append( m )
		n_vec.append( m / A )
		V_vec.append( m / rho )
		mu_m_vec.append( m * mu_m )
		f1_vec.append( f1 * m / A )
		f2_vec.append( f2 * m / A )

	V_tot = np.sum( V_vec )
	n_tot = np.sum( n_vec )
	rho = 1 / V_tot
	A = 1 / n_tot
	mu_m = np.sum( mu_m_vec )
	mu_l = mu_m * rho
	f1 = np.sum( f1_vec ) / n_tot
	f2 = np.sum( f2_vec ) / n_tot
		
	if raie == 'b':
		raie = 'Beta'
	elif raie == 'a':
		raie = 'Alpha'

	if affich == 1 or affich == 2 or affich == 3:
		print( u'\n----\nÉmetteur :\t\t' + emetteur + ' K-' + raie )
		print( 'Longueur d\'onde :\t' + str( lam ) +'\tAngstrom' )
		print( '\nAbsorbeur :\t\t' + mat[0][0] )
		print( 'Masse atomique :\t' + str( A ) +'\tg/mol' )
		print( u'Densité :\t\t' + str( rho ) + '\tg/cm3' )
		print( 'Coeff. abs. mass. :\t' + str( mu_m ) + '\tcm2/g' )
		print( 'Coeff. abs. lin. :\t' + str( mu_l ) + '\tcm-1')
		print( 'Facteurs de correction pour diffusion anormale :' )
		print( '\tf1 :\t\t' + str( f1 ) )
		print( '\tf2 :\t\t' + str( f2 ) )

	if affich == 2 or affich == 3:
		z=np.arange(0, 5/mu_l, 0.05/mu_l)
		I=np.exp( -mu_l*z )
		leg = emetteur + ' K-' + raie + ' dans ' + mat[0][0]  
		plt.plot( z, I, label = leg)
		plt.xlabel( 'Epaisseur (cm)' )
		plt.ylabel( 'Intensite transmise I/I0' )
		plt.legend( )
		if affich == 2:
			plt.show()

	return [mu_m, A, rho, lam, f1, f2]

def quant( spectre, liste_phases, affich = 1 ):
	"""

	Calcul le pourcentage volumique d'austénite. 
	Prend en compte que seulement deux phases existent : Austénite et ferrite (ou martensite).
	Utilise la méthodologie proposée dans le National Bureau of Standards [NBS Technical Note 709].

	Args :
		spectre : Spectre obtenu par l'algorithme xrdsim
		liste_phases : Liste des phases simulées obtenue par la fonction phase
		affich : Valeur par défaut de 1, affiche les résultats en texte à l'écran

	Méthodologie :	
		Pour chaque paire de plans (hkl)_alpha et (h'k'l')_gamma, on calcule V, la fraction volumique d'austénite, telle que :

		V = (R_alpha/A_alpha)_(hkl) / ( (R_alpha/A_alpha)_(hkl) + (R_gamma/A_gamma)_(h'k'l')

		A(hkl) = Intensité observée du pic (hkl). Correspond à l'aire sous la courbe de la PSF obtenue par l'algorithme xrdsim.
		R(hkl) = Facteur d'échelle de la réflexion (hkl)

		

	Formats :
		
		liste_phases_sim = dict : Liste des phases dans la simulation obtenue par xrdsim
			{phase_id, [A_tot, liste_pics] }
		
			liste_pics = list
				liste_pics[i] = [(h, k, l), theta, A]

		materiau = list
				Exemple de format de liste. 'B' signifie la balance
				[ [ 'Fe-0.4C'   , 'm'		],
				  [ 'C'		, 0.004		],
				  [ 'Fe'	, 'B'		] ]

		phase_list = dict
			{'MatBase', [mat, emetteur, raie]}
			{phase_id, [descr, cryst_struct, a, c, V, liste_pics]}

			liste_pics = list
				[liste_pics[0], liste_pics[1] ... ]

			liste_pics[i] = [(h, k, l), p, d, theta, FF, Lf, pf, Tf, R]

	"""
	 
	liste_phases_sim = {}

	mat = liste_phases['MatBase'][0]
	emetteur = liste_phases['MatBase'][1]
	raie = liste_phases['MatBase'][2]

	[mu_m, A, rho, lam, f1, f2] = read_melange( mat, emetteur, raie, affich )

	for i in range( len( spectre.peak_list ) ):
		phase_id = spectre.peak_list[i][4]
		if spectre.peak_list[i][1] == [] or spectre.peak_list[i][3] == False or phase_id == '' or type(spectre.peak_list[i][5]) != tuple:
			continue
		
		PSF = spectre.peak_list[i][1][0]
		if PSF == 'v':
			I0g = spectre.peak_list[i][1][1]
			I0l = spectre.peak_list[i][1][2]
			th0 = spectre.peak_list[i][1][3]/2.
			beta_g = spectre.peak_list[i][1][4]
			beta_l = spectre.peak_list[i][1][5]
			k_voigt = beta_l / (np.pi**0.5 * beta_g)
			A = beta_g*beta_l*I0g*I0l
			
		elif PSF == 'v2':
			I0lg = spectre.peak_list[i][1][1]
			th0 = spectre.peak_list[i][1][2]/2.
			beta_g = spectre.peak_list[i][1][3]
			beta_l = spectre.peak_list[i][1][4]
			k_voigt = beta_l / (np.pi**0.5 * beta_g)
			A = beta_g*beta_l*I0lg**2

		if not phase_id in liste_phases_sim:
			liste_phases_sim[phase_id] = [0, [] ]

		liste_phases_sim[phase_id][0] += A
		existe = False
		for j in range( len(liste_phases_sim[phase_id][1])):
			if liste_phases_sim[phase_id][1][j][0] == spectre.peak_list[i][5]:
				liste_phases_sim[phase_id][1][j][1] = min( liste_phases_sim[phase_id][1][j][1], th0 )
				liste_phases_sim[phase_id][1][j][2] += A
				existe = True
	
		if existe == False:
			liste_phases_sim[phase_id][1].append([spectre.peak_list[i][5], th0, A])

	print('\n----\n')
	for key in liste_phases_sim:
		if key == 'MatBase':
			continue

		if key not in liste_phases:
			print( u'La phase ' + key + u', contenue dans la simulation, n\'est pas dans la liste des phases du matériau' )
			continue


		cryst_struct = liste_phases[key][1]
		a = liste_phases[key][2]
		V = liste_phases[key][4]
		liste_pics_mat = liste_phases[key][5]


		print( 'Phase :\t\t\t' + key)
		print( 'Description :\t\t' + liste_phases[key][0] )
		
		print( 'Microstructure :\t' + cryst_struct )
		print( 'Param. de maille :\t' + str(a) + ' Angstroms' )
		print( 'Volume maille :\t\t' + str(round(V, 3)) + ' Angstroms^3' )
		
		print( u'\nListe des pics théoriques\n' )
		print( 'Plan\t   p\td (A)\tFF\tLP\tTF\tR' )
	
		R_tot = 0
		for i in range( len( liste_pics_mat ) ):
			hkl = liste_pics_mat[i][0]
			for j in range( len( liste_phases_sim[key][1] ) ):
				if liste_phases_sim[key][1][j][0] == hkl:
					R_tot += liste_pics_mat[i][8]
			print( str(liste_pics_mat[i][0]) + '  ' + str(liste_pics_mat[i][1]) + '\t' + str(round(liste_pics_mat[i][2], 4)) + '\t' + str(round(liste_pics_mat[i][4],1)) + '\t' + str(round(liste_pics_mat[i][5] * liste_pics_mat[i][6], 3)) + '\t' + str(round(liste_pics_mat[i][7], 4)) + '\t' + str(round(liste_pics_mat[i][8], 2)) )

		print( u'\nListe des pics observés\n' )
		print( u'Plan\t   2*theta\t2*theta\tÉcart\tR/R_tot A/A_tot\t  theo/obs  a_calc (A)' )
		print( u'\t   (obs.)\t(théo.)\t\t(théo.)\t(obs.)' )


		A_tot = liste_phases_sim[key][0]
		a_vec = []
	
		for i in range( len( liste_phases_sim[key][1] ) ):
			hkl = liste_phases_sim[key][1][i][0]
			th_theo = 0
			R = 0
			for j in range( len(liste_pics_mat) ):
				if liste_pics_mat[j][0] == hkl:
					th_theo = liste_pics_mat[j][3]*360/(2*np.pi)
					R = liste_pics_mat[j][8]
			th_obs = liste_phases_sim[key][1][i][1]		
			a_calc = calc_a( hkl, th_obs*2*np.pi/360, lam )
			a_vec.append(a_calc)
			A = liste_phases_sim[key][1][i][2]
			print( str(hkl) + '  ' + str(round( 2.*th_obs, 3 ) ) + '\t' + str(round( 2.*th_theo, 3 ) ) + '\t' + str(round( 2.*(th_obs-th_theo), 2 )) + '\t' + str( round( 100.*R/R_tot, 3)) + '\t' + str( round( 100.*A/A_tot, 3)) + '\t  ' + "%.3g" % ((R/R_tot)/(A/A_tot)) + '\t    ' + str(round(a_calc, 3)) )
		print( u'Paramètre de maille moyen calculé : ' + str(round(np.mean(a_vec), 3) ) )	

		print '\n----\n'

	A_tot_max = 0
	if len( liste_phases_sim ) == 2:
		for key in liste_phases_sim:
			A = liste_phases_sim[key][0] 
			if A > A_tot_max:
				A_tot_max = A
				phase1 = key
		for key in liste_phases_sim:
			if key != phase1:
				phase2 = key

		print( u'Austénite par paire de plans' )
		print( '\n' + phase1 + '\t\t' + phase2 + '\t\t' + phase2 +' (% vol.)\t' + phase1 + ' (% vol.)\t') # + 'R1\tI1\tR2\tI2' )
		R_I_1_vec = []
		R_I_2_vec = []

		for i in range(len(liste_phases_sim[phase1][1])):
			hkl1 = liste_phases_sim[phase1][1][i][0]
			I1 = liste_phases_sim[phase1][1][i][2]
			for j in range(len(liste_phases[phase1][5])):
				if liste_phases[phase1][5][j][0] == hkl1:
					R1 = liste_phases[phase1][5][j][8]

			for j in range(len(liste_phases_sim[phase2][1])):
				hkl2 = liste_phases_sim[phase2][1][j][0]
				I2 = liste_phases_sim[phase2][1][j][2]
				for k in range(len(liste_phases[phase2][5])):
					if liste_phases[phase2][5][k][0] == hkl2:
						R2 = liste_phases[phase2][5][k][8]

				R_I_1 = R1/I1
				R_I_2 = R2/I2
				V_2 = R_I_1 / (R_I_1 + R_I_2)
				
				R_I_1_vec.append(R_I_1)
				R_I_2_vec.append(R_I_2)
				
				print( str(hkl1) + '\t' + str(hkl2) + '\t' + str(round( 100* V_2, 2 )) + '\t\t' + str(round( (1-V_2)*100 , 2) ) + '\t\t') # + str(round(R1, 2)) + '\t' + str(round(I1, 2)) + '\t' +  str(round(R2,2)) + '\t' + str(round(I2,2))) 

		R_I_1_moy = np.mean( R_I_1_vec )
		R_I_2_moy = np.mean( R_I_2_vec )
		s_R_I_1 = np.std( R_I_1_vec, ddof=1 )
		s_R_I_2 = np.std( R_I_2_vec, ddof=1 )
		V_moy = R_I_1_moy / ( R_I_1_moy + R_I_2_moy )
		s_V = V_moy*(1-V_moy)*( (s_R_I_1/R_I_1_moy)**2 + (s_R_I_2/R_I_2_moy)**2)**0.5

		print( '\nFraction moyenne de ' + phase2 + ' :\t' + str(round(V_moy*100, 2)) + ' % vol.' )
		print( u'Écart-type :\t\t\t' + str(round(100*s_V, 2)) + ' % vol.' )


def list_var( spectre ):
	"""
	
	Dresse la liste des variables raffinées avec leur variance et écart-type.
	La matrice de covariance Vx est calculée par la formule :
		Vx = inv(transpose(J) * W * J)
		où J est la matrice jacobienne
		   W est la matrice de pondération : W[i, i] = 1/std[i]
		   std[i] est l'écart-type du compte au point i du vecteur des données brutes
		   std[i] = N
		   N est le nombre de comptes
	La matrice de covariance n'est pas sauvegardée par l'instruction 'spectre.write' donc elle doit être regénérée à chaque session de travail par une itération incluant toutes les valeurs à raffiner.

	La fonction imprime à l'écran les clés, les variables, les valeurs, les variances et les écarts-type.
	Les clés peuvent être utilisés pour la fonction 'corr_var', qui retourne les variances et écarts-types si désiré.

	
	"""
	
	if not spectre.etat.vardata:
		print( u'Faire une itération pour générer les données statistiques nécessaires. Ne pas oublier d\'inclure tous les pics et l\'arrière-plan' )
		return

	n = len( spectre.fit.Vx )
	pics_todate = []
	for i in range( n ):
		cle = spectre.fit.list_args[i]
		if cle[0] == 'p':
			pic = int( cle[1:-1] )
			if pic in pics_todate:
				continue
			else:
				pics_todate.append(pic)

			for j in range( len( spectre.peak_list ) ):
				if spectre.peak_list[j][0] == pic:
					pic_index = j
			
			print( "Pic # : %2d" % pic )
			PSF = spectre.peak_list[pic_index][1][0]

			if PSF == 'g' or PSF == 'l':
				I = spectre.peak_list[pic_index][1][1]
				var_I = spectre.fit.Vx[i, i]
				print(u'\tClé : %s\tI :	   %.2f; variance : %.2e; ecart-type : %.3f' % (cle, I, var_I, var_I**0.5) )
				th0 = spectre.peak_list[pic_index][1][2]
				var_th = spectre.fit.Vx[i+1, i+1]
				print('u\tClé : %s\ttheta_0 : %.3f; variance : %.2e; ecart-type : %.3e' % (cle, th0, var_th, var_th**0.5) )
				c = spectre.peak_list[pic_index][1][3]
				var_c = spectre.fit.Vx[i+2, i+2]
				print('\tc :	   %.2f; variance : %.2f; ecart-type : %.3f' % (c, var_c, var_c**0.5) )

			elif PSF == 'v':
				I0g = spectre.peak_list[pic_index][1][1]
				var_I0g = spectre.fit.Vx[i, i]
				print('\tI0g :     %.2f; variance : %.2e; ecart-type : %.3f' % (I0g, var_I0g, var_I0g**0.5) )
				I0l = spectre.peak_list[pic_index][1][2]
				var_I0l = spectre.fit.Vx[i+1, i+1]
				print('\tI0l :     %.2f; variance : %.2e; ecart-type : %.3f' % (I0l, var_I0l, var_I0l**0.5) )
				th0 = spectre.peak_list[pic_index][1][3]
				var_th = spectre.fit.Vx[i+2, i+2]
				print('\ttheta_0 : %.3f; variance : %.2e; ecart-type : %.3e' % (th0, var_th, var_th**0.5) )
				beta_g = spectre.peak_list[pic_index][1][4]
				var_bg = spectre.fit.Vx[i+3, i+3]
				print('\tbeta_g :  %.5f; variance : %.3e; ecart-type : %.3e' % (beta_g, var_bg, var_bg**0.5) )
				beta_l = spectre.peak_list[pic_index][1][5]
				var_bl = spectre.fit.Vx[i+4, i+4]
				print('\tbeta_l :  %.5f; variance : %.3e; ecart-type : %.3e' % (beta_l, var_bl, var_bl**0.5) )

			elif PSF == 'v2' or PSF[0:3] == 'v2k' or PSF[0:3] == 'v3k':
				I0lg = spectre.peak_list[pic_index][1][1]
				var_I0lg = spectre.fit.Vx[i, i]
				print(u'\tClé : p%d1\tI0lg :    %.2f; variance : %.2e; ecart-type : %.3f' % (pic, I0lg, var_I0lg, var_I0lg**0.5) )
				th0 = spectre.peak_list[pic_index][1][2]
				var_th = spectre.fit.Vx[i+1, i+1]
				print(u'\tClé : p%d2\ttheta_0 : %.3f; variance : %.2e; ecart-type : %.3e' % (pic, th0, var_th, var_th**0.5) )
				beta_g = spectre.peak_list[pic_index][1][3]
				var_bg = spectre.fit.Vx[i+2, i+2]
				print(u'\tClé : p%d3\tbeta_g :  %.5f; variance : %.3e; ecart-type : %.3e' % (pic, beta_g, var_bg, var_bg**0.5) )
				beta_l = spectre.peak_list[pic_index][1][4]
				var_bl = spectre.fit.Vx[i+3, i+3]
				print('\tClé : p%d4\tbeta_l :  %.5f; variance : %.3e; ecart-type : %.3e' % (pic, beta_l, var_bl, var_bl**0.5) )


def corr_var( spectre, cle1, cle2, affich = 1 ):
	"""

	Description à compléter

	"""
	if not spectre.etat.vardata:
		print( u'Faire une itération pour générer les données statistiques nécessaires. Ne pas oublier d\'inclure tous les pics et l\'arrière-plan' )
		return

	n = len( spectre.fit.Vx )
	for i in range( n ):
		cle = spectre.fit.list_args[i]
		if cle == cle1:
			var1_index = i
		if cle == cle2:
			var2_index = i

	for i in range( len( spectre.peak_list ) ):
		if spectre.peak_list[i][0] == int( cle1[1:-1] ):
			pic1_index = i
		if spectre.peak_list[i][0] == int( cle2[1:-1] ):
			pic2_index = i

	PSF1 = spectre.peak_list[pic1_index][1][0]
	if PSF1 == 'v2' or PSF1[0:3] == 'v2k' or PSF1[0:3] == 'v3k':
		arg = int(cle1[-1])
		if arg == 1:
			variable1 = 'I0lg'
		elif arg == 2:
			variable1 = 'theta0'
		elif arg == 3:
			variable1 = 'beta_g'
		elif arg == 4:
			variable1 = 'beta_l'
		
		valeur1 = spectre.peak_list[pic1_index][1][arg]
		var1 = spectre.fit.Vx[var1_index, var1_index]
		stdev1 = var1**0.5


	PSF2 = spectre.peak_list[pic2_index][1][0]
	if PSF2 == 'v2' or PSF2[0:3] == 'v2k' or PSF2[0:3] == 'v3k':
		arg = int(cle2[-1])
		if arg == 1:
			variable2 = 'I0lg'
		elif arg == 2:
			variable2 = 'theta0'
		elif arg == 3:
			variable2 = 'beta_g'
		elif arg == 4:
			variable2 = 'beta_l'
		
		valeur2 = spectre.peak_list[pic2_index][1][arg]
		var2 = spectre.fit.Vx[var2_index, var2_index]
		stdev2 = var2**0.5
	
	covar = spectre.fit.Vx[var1_index, var2_index]
	correl = covar/(stdev1*stdev2)

	if affich == 1:
		print( 'Variable 1 : ' )
		print( u'\tPic # : %d;\tVariable : %s;\tValeur : %.6e;\tVariance : %.3e;\tÉcart-type : %.3e' % (int(cle1[1:-1]), variable1, valeur1, var1, stdev1) )
		print( '\nVariable 2 : ' )
		print( u'\tPic # : %d;\tVariable : %s;\tValeur : %.6e;\tVariance : %.3e;\tÉcart-type : %.3e' % (int(cle2[1:-1]), variable2, valeur2, var2, stdev2) )
		print( '\nCovariance : %.3e' % covar )
		print( u'Corrélation : %.3f' % correl )

	v1 = [valeur1, var1]
	v2 = [valeur2, var2]
	return [v1, v2, covar]
