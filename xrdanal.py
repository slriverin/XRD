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
from scipy.optimize import curve_fit
import os.path
import csv
this_dir, this_filename = os.path.split( __file__ )
DATA_PATH = os.path.join( this_dir, "data" )

try:
	input = input
except:
	pass

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


def phase( phase_list, mat, phase_id, descr, cryst_struct, emetteur = 'Cu', raie = 'a', a_sig_a = (0.,0.), c_sig_c = (0.,0.), FV_sig_FV = (1., 0.), pourc_C = 0., strain_sig_strain = (0., 0.), alpha_sig_alpha = (13.3*2*np.pi/360, 4e-5), anom_scatt = True, affich = 0, liste_pics = [], liste_p = [], maj = False ):
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
		FV : Fraction volumique
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
			{phase_id, [descr, cryst_struct, a, c, V, liste_pics, mat, FV, strain, corr_ori]}

		liste_pics = list
			[liste_pics[0], liste_pics[1] ... ]

		liste_pics[i] = [(h, k, l), p, d, theta, FF, Lf, pf, Tf, R]

	"""

	if maj == True:
		phase = phase_list[phase_id]
		a_sig_a = phase[2]
		c_sig_c = phase[3]
		FV_sig_FV = phase[7]
		strain_sig_strain = phase[8]
		phase[7] = (FV_sig_FV[0], FV_sig_FV[1])
		corr_ori = phase[9]

	a, sig_a = a_sig_a
	c, sig_c = c_sig_c
	alpha, sig_alpha = alpha_sig_alpha
	FV, sig_FV = FV_sig_FV
	strain, sig_strain = strain_sig_strain
	
	if 'MatBase' in phase_list:
		if phase_list['MatBase'][1] != emetteur or phase_list['MatBase'][2] != raie:
			print(u'Veuillez vous assurer que la longueur d\'onde du rayon X incident sont les mêmes pour toutes les phases')
			return
	else:	
		phase_list['MatBase'] = [mat, emetteur, raie]
	
	if phase_id in phase_list and maj == False:
		ow = input( u'Phase déjà existante... écraser les données? 1 = oui, autre = non ... : '.encode('cp850') )
		if ow != '1':
			return

	FV_tot = 0
	var_FV_tot = 0
	for phase in phase_list:
		if phase == 'MatBase':
			continue
		FV_tot += phase_list[phase][7][0]
		var_FV_tot += phase_list[phase][7][1]
	
	if FV_tot + FV > 1 and maj == False:
		print( 'Total des fractions. vol > 1.' )
		print( '1. Normaliser' )
		print( u'2. Donner à cette phase la fraction disponible restante, soit : %.3f' %(1 - FV_tot) )
		print( '3 ou autre. Annuler' )
		print( u'L\'incertitude sera calculée à partir des incertitudes des autres fractions volumiques' )
		choix = input( 'Choix? ... : ' )
		if choix == '1':
			for phase in phase_list:
				if phase == 'MatBase':
					continue
				phase_list[phase][7] = ( phase_list[phase][7][0]/(FV + FV_tot), phase_list[phase][7][1]/(FV + FV_tot) )
			FV = FV / (FV + FV_tot)
			sig_FV = var_FV_tot**0.5/(FV + FV_tot)
		elif choix == '2':
			FV = 1 - FV_tot
			sig_FV = var_FV_tot**0.5
		else:
			return

	elif FV_tot > 1 and maj == True:	
		for phase in phase_list:
			if phase == 'MatBase':
				continue
			phase_list[phase][7] = ( phase_list[phase][7][0]/FV_tot, phase_list[phase][7][1]/FV_tot )
			FV = FV / FV_tot
			sig_FV = var_FV_tot**0.5/FV_tot




	[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = read_melange( mat, emetteur, raie, affich )

	if anom_scatt == True: #Facteurs de correction pour dispersion "anormale". Ici pour Cu K-alpha dans Fe [ITC vol. C table 4.2.6.8]
		if affich == 1:
			print( u'\nCorrection pour dispersion anormale :' )
			print( u'\tRéel : \t\tf1 = ' + str(f1) )
			print( u'\tImaginaire :	f2 = ' + str(f2) )

	else:
		(f1, sig_f1) = (0,0)
		(f2, sig_f2) = (0,0)

	if cryst_struct == 'BCC':	#Cubique centré
		if a == 0 :
			a = 2.872 #Par défaut, on attribue la valeur usuelle du paramètre de maille du fer alpha
			sig_a = 0.001 #Valeur approximative à préciser
		
		c = 0;
		V = a**3
		fV = 1./V**2
		sig_fV = 6./a**7 * sig_a
		if maj == False:
			corr_ori = {}
			corr_ori['C410'] = (0., 0.)
			corr_ori['C610'] = (0., 0.)
			corr_ori['C810'] = (0., 0.)
		
		if liste_pics == []:
			liste_pics = [ [(1, 1, 0)], [(2, 0, 0)], [(2, 1, 1)], [(2, 2, 0)] ]
		if liste_p == []:
			liste_p = [12, 6, 24, 12]

		for i in range(len(liste_pics)):
			(h, k, l) = liste_pics[i][0]
			d = calc_d( liste_pics[i][0], a )
			sig_d = sig_a / (h**2 + k**2 + l**2)**0.5
			theta = np.arcsin( lam / (2*d) )
			var_theta = (sig_lam**2 + lam**2/d**2*sig_d**2)/(4*d**2*(1-lam**2/(4*d**2)))
			sig_theta = var_theta**0.5
			s = np.sin(theta) / lam
			sig_s = sig_d/(2*d**2)
			sf0, sig_sf0 = scatt_f( s, mat )
			sF = sf0 + f1 + 1j*f2
			FF = (4*sF*np.conj(sF)).real
			sig_FF = 4*(4*sf0**2*sig_sf0**2 + 4*f1**2*sig_f1**2 + 4*f2**2*sig_f2**2 + 4*sf0**2*f1**2*(sig_sf0**2/sf0**2 + sig_f1**2/f1**2) )**0.5
			
			p = liste_p[i]
			Lf, sig_Lf = lor_f( (theta, sig_theta) )
			pf, sig_pf = pol_f( (theta, sig_theta), (alpha, sig_alpha) )
			Tf, sig_Tf = temp_f( (s, sig_s) )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			sig_R = R*(sig_fV**2/fV**2 + sig_FF**2/FF**2 + sig_Lf**2/Lf**2 + sig_pf**2/pf**2 + sig_Tf**2/Tf**2)**0.5
			liste_pics[i].append(p)
			liste_pics[i].append((d, sig_d))
			liste_pics[i].append((theta, sig_theta))
			liste_pics[i].append((FF, sig_FF))
			liste_pics[i].append((Lf, sig_Lf))
			liste_pics[i].append((pf, sig_pf))
			liste_pics[i].append((Tf, sig_Tf))
			liste_pics[i].append((R, sig_R))
		
	elif cryst_struct == 'FCC': 	#Cubique face centrée
		if a == 0:
			a = 3.555 + 0.044*pourc_C
			sig_a = 0.001
		c = 0;
		V = a**3
		fV = 1./V**2
		sig_fV = 6./a**7 * sig_a
		if maj == False:
			corr_ori = {}
			corr_ori['C410'] = (0., 0.)
			corr_ori['C610'] = (0., 0.)
			corr_ori['C810'] = (0., 0.)

		if liste_pics == []:
			liste_pics =  [ [(1, 1, 1)], [(2, 0, 0)], [(2, 2, 0)], [(3, 1, 1)], [(2, 2, 2)], [(3, 3, 1)], [(4, 0, 0)] ]
		if liste_p == []:
			liste_p = [8, 6, 12, 24, 8, 24, 24]

		for i in range(len(liste_pics)):
			(h, k, l) = liste_pics[i][0]
			d = calc_d( liste_pics[i][0], a )
			sig_d = sig_a / (h**2 + k**2 + l**2)**0.5
			theta = np.arcsin( lam / (2*d) )
			var_theta = (sig_lam**2 + lam**2/d**2*sig_d**2)/(4*d**2*(1-lam**2/(4*d**2)))
			sig_theta = var_theta**0.5
			s = np.sin(theta) / lam
			sig_s = sig_d/(2*d**2)
			sf0, sig_sf0 = scatt_f( s, mat )
			sF = sf0 + f1 + 1j*f2
			FF = (16*sF*np.conj(sF)).real
			sig_FF = 16*(4*sf0**2*sig_sf0**2 + 4*f1**2*sig_f1**2 + 4*f2**2*sig_f2**2 + 4*sf0**2*f1**2*(sig_sf0**2/sf0**2 + sig_f1**2/f1**2) )**0.5

			p = liste_p[i]
			Lf, sig_Lf = lor_f( (theta, sig_theta) )
			pf, sig_pf = pol_f( (theta, sig_theta), (alpha, sig_alpha) )
			Tf, sig_Tf = temp_f( (s, sig_s) )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			sig_R = R*(sig_fV**2/fV**2 + sig_FF**2/FF**2 + sig_Lf**2/Lf**2 + sig_pf**2/pf**2 + sig_Tf**2/Tf**2)**0.5
			liste_pics[i].append(p)
			liste_pics[i].append((d, sig_d))
			liste_pics[i].append((theta, sig_theta))
			liste_pics[i].append((FF, sig_FF))
			liste_pics[i].append((Lf, sig_Lf))
			liste_pics[i].append((pf, sig_pf))
			liste_pics[i].append((Tf, sig_Tf))
			liste_pics[i].append((R, sig_R))

	elif cryst_struct == 'DC':	#Diamond Cubic
		if a == 0:
			a = 5.43088
			sig_a = 4e-5	#Silicium selon CIF 27-1402

		c = 0;
		V = a**3
		fV = 1./V**2
		sig_fV = 6./a**7 * sig_a
		if maj == False:
			corr_ori = {}
			corr_ori['C410'] = (0., 0.)
			corr_ori['C610'] = (0., 0.)
			corr_ori['C810'] = (0., 0.)

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
			sig_d = sig_a / (h**2 + k**2 + l**2)**0.5
			theta = np.arcsin( lam / (2*d) )
			var_theta = (sig_lam**2 + lam**2/d**2*sig_d**2)/(4*d**2*(1-lam**2/(4*d**2)))
			sig_theta = var_theta**0.5
			s = 1./(2*d)
			sig_s = sig_d/(2*d**2)
			sf0, sig_sf0 = scatt_f( s, mat )
			sF = sf0 + f1 + 1j*f2
			if (h + k + l) % 2 == 0:	#Somme des indices paire
				FFmult = 64
			else:
				FFmult = 32
			FF = FFmult*(sF*np.conj(sF)).real
			sig_FF = FFmult * (4*sf0**2*sig_sf0**2 + 4*f1**2*sig_f1**2 + 4*f2**2*sig_f2**2 + 4*sf0**2*f1**2*(sig_sf0**2/sf0**2 + sig_f1**2/f1**2) )**0.5
			p = liste_p[i]
			Lf, sig_Lf = lor_f( (theta, sig_theta) )
			pf, sig_pf = pol_f( (theta, sig_theta), (alpha, sig_alpha) )
			Tf, sig_Tf = temp_f( (s, sig_s) )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			sig_R = R*(sig_fV**2/fV**2 + sig_FF**2/FF**2 + sig_Lf**2/Lf**2 + sig_pf**2/pf**2 + sig_Tf**2/Tf**2)**0.5
			liste_pics[i].append(p)
			liste_pics[i].append((d, sig_d))
			liste_pics[i].append((theta, sig_theta))
			liste_pics[i].append((FF, sig_FF))
			liste_pics[i].append((Lf, sig_Lf))
			liste_pics[i].append((pf, sig_pf))
			liste_pics[i].append((Tf, sig_Tf))
			liste_pics[i].append((R, sig_R))

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

	elif cryst_struct == 'BCT':	#Quadratique centré
		V = a**2*c

		if liste_pics == []:
			liste_pics = [	[(1, 0, 1)], [(1, 1, 0)], [(0, 0, 2)], [(2, 0, 0)], [(1, 1, 2)], 
				      	[(2, 1, 1)], [(2, 0, 2)], [(2, 2, 0)], [(1, 0, 3)], [(3, 0, 1)],
					[(3, 1, 0)], [(2, 2, 2)] ]
		if liste_p == []:	
			liste_p = [	8, 4, 2, 4, 8,
					16, 8, 4, 8, 8,
					8, 8 ]

		for i in range( len(liste_pics) ):
			(h, k, l) = liste_pics[i][0]
			d = calc_d( liste_pics[i][0], a, c, 'q' )
			sig_d = d**3 * ( (h**2 + k**2)**2/a**6 * sig_a**2 + l**4/c**6*sig_c**2)**0.5
			theta = np.arcsin( lam / (2*d) )
			var_theta = (sig_lam**2 + lam**2/d**2*sig_d**2)/(4*d**2*(1-lam**2/(4*d**2)))
			sig_theta = var_theta**0.5
			s = 1./(2*d)
			sig_s = sig_d/(2*d**2)
			sf0, sig_sf0 = scatt_f( s, mat )
			sF = sf0 + f1 + 1j*f2
			FF = 4*(sF*np.conj(sF)).real
			sig_FF = 4 * (4*sf0**2*sig_sf0**2 + 4*f1**2*sig_f1**2 + 4*f2**2*sig_f2**2 + 4*sf0**2*f1**2*(sig_sf0**2/sf0**2 + sig_f1**2/f1**2) )**0.5
			p = liste_p[i]
			Lf, sig_Lf = lor_f( (theta, sig_theta) )
			pf, sig_pf = pol_f( (theta, sig_theta), (alpha, sig_alpha) )
			Tf, sig_Tf = temp_f( (s, sig_s) )
			R = 1./V**2 * FF*p*Lf*pf*Tf
			sig_R = R*(sig_FV**2/FV**2 + sig_FF**2/FF**2 + sig_Lf**2/Lf**2 + sig_pf**2/pf**2 + sig_Tf**2/Tf**2)**0.5
			liste_pics[i].append(p)
			liste_pics[i].append((d, sig_d))
			liste_pics[i].append((theta, sig_theta))
			liste_pics[i].append((FF, sig_FF))
			liste_pics[i].append((Lf, sig_Lf))
			liste_pics[i].append((pf, sig_pf))
			liste_pics[i].append((Tf, sig_Tf))
			liste_pics[i].append((R, sig_R))

	#Calcul de la composition chimique globale de l'échantillon
	phase_list[phase_id] = [descr, cryst_struct, (a, sig_a), (c, sig_c), V, liste_pics, mat, (FV, sig_FV), (strain, sig_strain), corr_ori]
	MatBase = [['MatBase', 'm']]
	w_tot = 0.
	m_tot = 0.
	for phase in phase_list:
		if phase == 'MatBase':
			continue
		phase_list[phase][6] = mat_conv(phase_list[phase][6])
		FV, sig_FV = phase_list[phase][7]
		[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = read_melange( mat, emetteur, raie, affich )
		m_tot += FV*rho

	for phase in phase_list:
		if phase == 'MatBase':
			continue
		phase_list[phase][6] = mat_conv(phase_list[phase][6])
		FV, sig_FV = phase_list[phase][7]
		[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = read_melange( mat, emetteur, raie, affich )
		w_phase = FV*rho/m_tot
		sig_w_phase = (1. - w_phase)/m_tot*(rho**2*sig_FV**2 + FV**2*sig_rho**2)**0.5

		for i in range(len(phase_list[phase][6])):
			if i == 0:
				continue
			elem = phase_list[phase][6][i][0]
			ajoute_elem = True
			for j in range(len(MatBase)):
				if j == 0:
					continue
				if elem == MatBase[j][0]:
					w_el_ph, sig_w_el_ph = read_nmbr( phase_list[phase][6][i][1] )
					w_el_mb, sig_w_el_mb = read_nmbr( MatBase[j][1] )
					MatBase[j][1] = write_nmbr( (w_el_ph * w_phase + w_el_mb, (w_phase**2*sig_w_el_ph**2 + w_el_ph**2*sig_w_phase**2 + sig_w_el_mb**2)**0.5 ) )
					ajoute_elem = False

			if ajoute_elem == True:
				w_el_ph, sig_w_el_ph = read_nmbr( phase_list[phase][6][i][1] )
				w_text = write_nmbr( (w_el_ph * w_phase, (w_phase**2*sig_w_el_ph**2 + w_el_ph**2*sig_w_phase**2)**0.5 ) )
				MatBase.append( [elem, w_text ] )
				
	phase_list['MatBase'][0] = MatBase 

					

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
		R = phase_data[5][i][8][0]
		if R > R_max:
			R_max = R
	
	for i in range( len( phase_data[5] ) ):
		angle_2theta=[0,0]
		angle_2theta[0] = 2*phase_data[5][i][3][0] * 360. / (2.*np.pi)
		angle_2theta[1] = 2*phase_data[5][i][3][1] * 360. / (2.*np.pi)
		if raw_data != [] and ( angle_2theta > theta_max or angle_2theta < theta_min ):
			continue
		R=[0,0]
		R[0] = phase_data[5][i][8][0] / R_max
		R[1] = phase_data[5][i][8][1] / R_max
		plt.plot([angle_2theta[0], angle_2theta[0]], [0, R[0]], color = couleur)
		plt.errorbar( angle_2theta[0], R[0], xerr = angle_2theta[1], yerr = R[1], fmt = 'o', color = couleur, ecolor = 'g')
		plt.annotate( str( phase_data[5][i][0] ), [angle_2theta[0], 0.5 * R[0]] )

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
			'q' : Quadratique

	"""	
	
	h = hkl[0]
	k = hkl[1]
	l = hkl[2]
	
	if cryst_syst == 'c':
		return np.sqrt( a**2 / (h**2 + k**2 + l**2) )
	elif cryst_syst == 'h':
		return ( 4./3.*(h**2 + h*k + k**2)/a**2 + l**2/c**2 )**(-0.5)
	elif cryst_syst == 'q':
		return ( (h**2 + k**2)/a**2 + l**2/c**2 )**-0.5

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
		err = np.array([])

		for j in range( len(s) ):
			s_f_temp, err_temp = scatt_f( s[j], mat )
			s_f = np.append( s_f, s_f_temp )
			sig_sf = np.append( err, err_temp )

	#Ici, un seul angle est étudié
	elif type(s) == int or type(s) == float or type(s) == np.float64:

		#S'il s'agit d'un mélange de plusieurs éléments, une récursion est faite pour trouver le facteur moyen pondéré
		#selon la fraction atomique
		if type( mat ) == list:
			mat_conv( mat, 'a' )
			s_f_vect = []
			err_vect = []
			for i in range( len(mat) - 1):
				mat_temp = mat[i+1][0]
				n, sig_n = read_nmbr(mat[i+1][2])
				s_f_temp, err_temp = scatt_f( s, mat_temp )
				s_f_vect.append( s_f_temp * n )
				err_vect.append( err_temp * n )
			
			s_f = np.sum( s_f_vect )
			sig_sf = np.sum( err_vect )

		#Un seul élément, substance pure
		elif type( mat ) == str:
			X = read_el( mat )
			a = X[6]
			b = X[7]
			c = X[8]
			sig_sf = X[11]

			s_f = c
			for i in range(4):
				s_f += a[i]*np.exp( -b[i]*s**2 )

			if s > 2: 
				print( u'Attention : s > 2, corrélation imprécise' )
				
	return (s_f, sig_sf)


def pol_f( theta_sig_theta, alpha_sig_alpha = (13.3*2*np.pi/360, 4e-5) ):
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

	theta, sig_theta = theta_sig_theta
	alpha, sig_alpha = alpha_sig_alpha
	pf =  (1. + np.cos(2.*alpha)**2. * np.cos(2.*theta)**2. ) /  (1. + np.cos(2.*alpha)**2. )
	k = (np.cos(2*alpha))**2
	sig_k = 4*np.cos(2*alpha)*np.sin(2*alpha)*sig_alpha
	dpfdk = (1 + (np.cos(2*theta))**2)/(1+k) + (1 + k*(np.cos(2*theta))**2)/(1+k)**2
	dpfdtheta = 4*k/(1+k)*np.cos(2*theta)*np.sin(2*theta)
	sig_pf = (dpfdk**2*sig_k**2 + dpfdtheta**2*sig_theta**2)**0.5
	return (pf, sig_pf)


def lor_f( theta_sig_theta ):
	"""
	Fonction calculant le facteur de Lorentz [ITC vol. C, sec. 6.2.5]

	Lf = 1 / ( sin^2(theta)*cos(theta) )

	Arg :
		theta = angle de diffraction de l'échantillon

	"""

	theta, sig_theta = theta_sig_theta
	Lf = 1 / (np.sin(theta)**2 * np.cos(theta))
	sig_Lf = (-2/(np.sin(theta))**3 + 1/(np.sin(theta)*(np.cos(theta))**2))*sig_theta
	return (Lf, sig_Lf)

def temp_f( s_sig_s, mat = 'Fe', T_sig_T = (293., 1) ):
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

	s, sig_s = s_sig_s
	T, sig_T = T_sig_T

	if type( mat ) == str:
		X = read_el( mat )
		DebyeT, sig_DebyeT = X[5]
		A, sig_A = X[2]
		x = DebyeT / T
		sig_x = (sig_DebyeT**2/T**2 + DebyeT**2/T**4*sig_T**2)**0.5
		c = phiofx(x) + x/4
		sig_c = ((-phiofx(x)/x + 1/(np.exp(x)-1))**2 + 1./16)*sig_x
		M = 1.1490292193e4 * T / (A*DebyeT**2) * c * s**2
		sig_M = M * (sig_T**2/T**2 + 4*sig_DebyeT**2/DebyeT**2 + sig_c**2/c**2 + 4*sig_s**2/s**2 + sig_A**2/A**2)**0.5
		Tf = np.exp(-2*M)
		sig_Tf = 2*np.exp(-2*M)*sig_M
		return (Tf, sig_Tf)

	elif type( mat ) == list:
		mat_conv( mat, 'a' )
		T_f_vect = []
		for i in range( len(mat) - 1):
			mat_temp = mat[i+1][0]
			n = mat[i+1][1]
			T_f_vect.append( temp_f( (s, sig_s), mat_temp, (T, sig_T) ) * n )
		
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





def read_el( absorbeur, emetteur = 'Cu', raie = 'a', affich = 0, ret_incert = False ):
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
		Les valeurs sont retournées avec leur incertitude

	"""
	#Choisit la bonne colonne pour la lecture des informations
	if emetteur == 'Mo':
		rowno = 4
	elif emetteur == 'Cu':
		rowno = 6
		rowno2 = 23
	elif emetteur == 'Co':
		rowno = 8
		rowno2 = 25
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
		print( DATA_PATH )
		return

	with open( os.path.join( DATA_PATH, 'Coeff_abs.csv' ), 'r', encoding='latin-1' ) as csvfile:
		csvreader = csv.reader( csvfile, delimiter = ',' )
		for row in csvreader:
			if row[0] == 'Lambda':
				lam = read_nmbr( row[rowno] )
			elif row[0] == absorbeur:
				Z = int( row[1] )
				A = read_nmbr( row[2] )
				rho = read_nmbr( row[3] )
				mu_m = read_nmbr( row[rowno] )
				DebyeT = read_nmbr( row[12] )
				a = [ float( row[13] ), float( row[14] ), float( row[15] ), float( row[16] ) ]
				b = [ float( row[17] ), float( row[18] ), float( row[19] ), float( row[20] ) ]
				c = float( row[21] )
				incertsf = float( row[22] )
				f1 = read_nmbr( row[rowno2] )
				f2 = read_nmbr( row[rowno2 + 1] )
				mu_l_val = mu_m[0]*rho[0]
				mu_l_sd = mu_l_val*( (mu_m[1]/mu_m[0])**2 + (rho[1]/rho[0])**2 )**0.5
				mu_l = (mu_l_val, mu_l_sd)
				if affich == 1 or affich == 2 or affich == 3:
					print( u'\n----\nÉmetteur :\t\t' + emetteur + ' K-' + raie )
					print( 'Longueur d\'onde :\t' + write_nmbr(lam) +'\tAngstrom' )
					print( '\n' )		
					print( 'Absorbeur :\t\t' + absorbeur )
					print( u'Numéro atomique :\t' + str( Z ) )
					print( 'Masse atomique :\t' + write_nmbr( A ) + '\tg/mol' )
					print( u'Densité :\t\t' + write_nmbr( rho ) + '\tg/cm3' )
					print( 'Coeff. abs. mass. :\t' + write_nmbr( mu_m ) + '\tcm2/g' )
					print( 'Coeff. abs. lin. :\t' + write_nmbr( mu_l ) + '\tcm-1')
					print( 'Temp. de Debye :\t' + write_nmbr( DebyeT ) + '\tK' )
					print( 'Facteurs de correction pour diffusion anormale :' )
					print( '\tf1 :\t\t' + write_nmbr( f1 ) )
					print( '\tf2 :\t\t' + write_nmbr( f2 ) )

				if affich == 2 or affich == 3:
					z=np.arange(0, 5/mu_l[0], 0.05/mu_l[0])
					I=np.exp( -mu_l[0]*z )
					leg = emetteur + ' K-' + raie + ' dans ' + absorbeur  
					plt.plot( z, I, label = leg)
					plt.xlabel( 'Epaisseur (cm)' )
					plt.ylabel( 'Intensite transmise I/I0' )
					plt.legend( )
					if affich == 2:
						plt.show()
				return [mu_m, Z, A, rho, lam, DebyeT, a, b, c, f1, f2, incertsf]

	print( 'Erreur, absorbeur inexistant' )


def mat_conv( mat, affich = 0 ):
	
	"""
	Complète les informations sur un matériau en calculant la balance et les concentrations atomiques
	Les caractéristiques du matériau doivent être contenues dans le fichier de donnees
	Intrants :
		mat :		Composition chimique du materiau. Format de type list. Voir ci-dessous
		type_fract :	Type de fraction voulue ('m' = massique, 'a' = atomique)

	Extrants :
		mat_conv :	Composition chimique convertie

	Formats :
		materiau :	Exemple de format de liste accepté. 'B' signifie la balance
				[ [ 'Fe-0.4C'   , 'm'		],
				  [ 'C'		, 0.0040(5)	],
				  [ 'Fe'	, 'B'		] ]

		matériau (retourné) :
				[ [ 'Fe-0.4C'   , 'm'		, 'a'			],
				  [ 'C'		, 0.0040(5)	, 1.83(22)e-02		],
				  [ 'Fe'	, 9.9600(50)e-01, 9.816696(90)e-01	] ]
	"""

	if len( mat[0] ) == 3:
		if affich == 1:
			print( u'Déjà dans le bon format' )
		
		return mat

	tot = 0.
	var_B = 0.
	index_B = 0
	for i in range( len(mat) - 1 ):
		if mat[i + 1][1] == 'B':
			index_B = i + 1
		else:
			x, sig_x = read_nmbr( mat[i + 1][1] )
			tot += x
			var_B += sig_x**2
	if index_B != 0:
		val_B = 1 - tot	
		mat[index_B][1] = write_nmbr( val_B, var_B**0.5 )

	if mat[0][1] == 'm':
		#Conversion qté massique en qté atomique
		tot = 0.
		n_vec = []
		for i in range( len(mat) - 1 ):
			if mat[i + 1][1] == 'B':
				mat = mat_conv( mat, 'm' )
			
			Ai, sig_Ai = read_el( mat[i + 1][0], 'Cu', 'a')[2]
			wi, sig_wi = read_nmbr( mat[i + 1][1] )
			n_vec.append( wi / Ai )

		Amel = 1./sum(n_vec)
		for i in range( len( n_vec ) ):
			ni = n_vec[i]*Amel
			Ai, sig_Ai = read_el( mat[i + 1][0], 'Cu', 'a')[2]
			wi, sig_wi = read_nmbr( mat[i + 1][1] )
			var_ni = (1. - ni)**2/Ai**2*(Amel**2*sig_wi**2 + ni**2*sig_Ai**2)
			mat[i + 1] = [ mat[i + 1][0], mat[i + 1][1], write_nmbr( ni, var_ni**0.5 ) ]

		mat[0] = [mat[0][0], 'm', 'a']
		return mat

	elif mat[0][1] == 'a':
		tot = 0.
		w_vec = []
		for i in range( len(mat) - 1 ):
			if mat[i + 1][1] == 'B':
				mat = mat_conv( mat, 'a' )
			
			Ai, sig_Ai = read_el( mat[i + 1][0], 'Cu', 'a')[2]
			ni, sig_ni = read_nmbr( mat[i + 1][1] )
			w_vec.append( ni * Ai )

		Amel = sum(w_vec)
		for i in range( len( w_vec ) ):
			wi = w_vec[i]/Amel
			Ai, sig_Ai = read_el( mat[i + 1][0], 'Cu', 'a')[2]
			ni, sig_ni = read_nmbr( mat[i + 1][1] )
			var_wi = (1. - wi)**2/Amel**2*(Ai**2*sig_ni**2 + ni**2*sig_Ai**2)
			mat[i + 1] = [ mat[i + 1][0], write_nmbr( ni, var_ni**0.5 ), mat[i + 1][1] ]

		mat[0] = [mat[0][0], 'm', 'a']
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
		Les valeurs et les écarts-types sont retournés
	
	"""
	mat_conv( mat, 'm' )
	V_vec = []
	n_vec = []
	w_vec = []
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

		w, sig_w = read_nmbr( mat[i + 1][1] )
		n_vec.append( w / A[0] )
		V_vec.append( w / rho[0] )
		mu_m_vec.append( w * mu_m[0] )
		f1_vec.append( f1[0] * w / A[0] )
		f2_vec.append( f2[0] * w / A[0] )

	V_tot = np.sum( V_vec )
	n_tot = np.sum( n_vec )
	rho_mel = 1 / V_tot
	A_mel = 1 / n_tot
	mu_m_mel = np.sum( mu_m_vec )
	mu_l_mel = mu_m_mel * rho_mel
	f1_mel = np.sum( f1_vec ) / n_tot
	f2_mel = np.sum( f2_vec ) / n_tot

	var_rho_vec = []
	var_A_vec = []
	var_mu_m_vec = []
	var_f1_mel_vec =[]
	var_f2_mel_vec =[]

	for i in range( len(mat) - 1):
		X = read_el( mat[i + 1][0], emetteur, raie )
		mu_mi, sig_mu_mi = X[0]
		Zi = X[1]
		Ai, sig_Ai = X[2]
		rhoi, sig_rhoi = X[3]
		lam, sig_lam = X[4]
		f1i, sig_f1i = X[9]
		f2i, sig_f2i = X[10]

		wi, sig_wi = read_nmbr( mat[i + 1][1] )
		var_rho_vec.append( rho_mel**4*(wi**2*sig_rhoi**2/rhoi**4 + sig_wi**2/rhoi**2 ) )
		var_A_vec.append( A_mel**4*(wi**2*sig_Ai**2/Ai**4 + sig_wi**2/rhoi**2) )
		var_mu_m_vec = wi**2 * sig_mu_mi**2 + mu_mi**2 * sig_wi**2
		var_f1_mel_vec.append( (f1_mel - f1i)**2/Ai**2 * (sig_wi**2 + wi**2/Ai**2*sig_Ai**2) )
		var_f2_mel_vec.append( (f2_mel - f2i)**2/Ai**2 * (sig_wi**2 + wi**2/Ai**2*sig_Ai**2) )
	
	var_rho_mel = np.sum( var_rho_vec )			
	var_A_mel = np.sum( var_A_vec )
	var_mu_m_mel = np.sum( var_mu_m_vec )
	var_f1_mel = np.sum( var_f1_mel_vec )
	var_f2_mel = np.sum( var_f2_mel_vec )
	var_mu_l_mel = mu_l_mel**2*( var_mu_m_mel/mu_m_mel**2 + var_rho_mel/rho_mel**2 )

	mu_m = (mu_m_mel, var_mu_m_mel**0.5)
	mu_l = (mu_l_mel, var_mu_l_mel**0.5)
	A = (A_mel, var_A_mel**0.5)
	rho = (rho_mel, var_rho_mel**0.5)
	f1 = (f1_mel, var_f1_mel**0.5)
	f2 = (f2_mel, var_f2_mel**0.5)
		
	if raie == 'b':
		raie = 'Beta'
	elif raie == 'a':
		raie = 'Alpha'

	if affich == 1 or affich == 2 or affich == 3:
		print( u'\n----\nÉmetteur :\t\t' + emetteur + ' K-' + raie )
		print( 'Longueur d\'onde :\t' + write_nmbr( lam, sig_lam ) +'\tAngstrom' )
		print( '\nAbsorbeur :\t\t' + mat[0][0] )
		print( 'Masse atomique :\t' + write_nmbr( A ) +'\tg/mol' )
		print( u'Densité :\t\t' + write_nmbr( rho ) + '\tg/cm3' )
		print( 'Coeff. abs. mass. :\t' + write_nmbr( mu_m ) + '\tcm2/g' )
		print( 'Coeff. abs. lin. :\t' + write_nmbr( mu_l ) + '\tcm-1')
		print( 'Facteurs de correction pour diffusion anormale :' )
		print( '\tf1 :\t\t' + write_nmbr( f1 ) )
		print( '\tf2 :\t\t' + write_nmbr( f2 ) )

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

	return [mu_m, A, rho, (lam, sig_lam), f1, f2]

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

		print( '\n----\n')

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

	Affiche des données comparatives sur deux variables de régression.

	Intrants :
		spectre : source des données
		cle1, cle2 : clés des variables à comparer
		affich = 1 : Affiche à l'écran les variables, les valeurs, les variances, les écarts-types, la covariance et la corrélation

	Extrants :	
		[ [valeur1, variance1],
		  [valeur2, variance2],
		  covariance		]

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

def constr_Vx( spectre, liste_cles ):
	"""

	Construit la matrice de variance-covariance des clés entrées dans la liste des clés.

	Intrants :
		spectre : source des données
		liste_cles = [cle1, cle2, ... ] : liste des clés

	Extrants :
		x = [x1, x2, ... ] : vecteur contenant les valeurs des variables de liste_cles
		Vx = Matrice de variance-covariance des variables

	"""
	
	if not spectre.etat.vardata:
		print( u'Faire une itération pour générer les données statistiques nécessaires. Ne pas oublier d\'inclure tous les pics et l\'arrière-plan' )
		return

	n = len( spectre.fit.Vx )
	m = len( liste_cles )
	
	Vx = np.zeros( (m, m) )
	x = np.zeros( m )
	for i in range( m ):
		A = corr_var( spectre, liste_cles[i], liste_cles[i], affich=0 )
		x[i] = A[0][0]
		Vx[i][i] = A[0][1]
		for j in range( m ):
			if i == j:
				continue

			A = corr_var( spectre, liste_cles[i], liste_cles[j], affich=0 )
			Vx[i][j] = A[2]
			
	return x, Vx		

def param_anal( spectre, phase_id, param, dict_phases = {}, affich = 1 ):
	"""

	Analyse et trace les données sélectionnées par l'utilisateur avec les incertitudes correspondantes

	Intrants :
		spectre : Source des données de régression
		phase_id : Identification de la phase analysée
		param : Données à analyser. Choix possibles : 'I0lg', 'beta_g', 'beta_l', 'k', 'A', 'I', 'beta', 'FWHM', 'form_fact', 'A/R', 'delta_th'
		dict_phases : Dictionnaire des phases requis pour certaines analyses

	Extrants :
		x, y, x_err, y_err : Valeurs x et y avec écarts-types respectifs

	"""

	x = []
	y = []
	x_err = []
	y_err = []
	if param == 'I0lg':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$I_{0lg}$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				x.append(spectre.peak_list[i][1][2])
				y.append(spectre.peak_list[i][1][1])
				x_cle = 'p' + str(pic) + '2'
				y_cle = 'p' + str(pic) + '1'
				A = corr_var( spectre, x_cle, y_cle, affich=0 )
				x_err.append( A[0][1]**0.5 )
				y_err.append( A[1][1]**0.5 )

	elif param == 'beta_g':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$\beta_{g}$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				x.append(spectre.peak_list[i][1][2])
				y.append(spectre.peak_list[i][1][3])
				x_cle = 'p' + str(pic) + '2'
				y_cle = 'p' + str(pic) + '3'
				A = corr_var( spectre, x_cle, y_cle, affich=0 )
				x_err.append( A[0][1]**0.5 )
				y_err.append( A[1][1]**0.5 )

	elif param == 'beta_l':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$\beta_{l}$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				x.append(spectre.peak_list[i][1][2])
				y.append(spectre.peak_list[i][1][4])
				x_cle = 'p' + str(pic) + '2'
				y_cle = 'p' + str(pic) + '4'
				A = corr_var( spectre, x_cle, y_cle, affich=0 )
				x_err.append( A[0][1]**0.5 )
				y_err.append( A[1][1]**0.5 )
	
	elif param == 'k':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$k$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_th, cle_bg, cle_bl] )
				beta_g = X[1]
				beta_l = X[2]
				k = beta_l/(beta_g*np.pi**0.5)
				J = np.array([0, -k/beta_g, k/beta_l])
				var_k = np.matmul( np.matmul( J, Vx ), np.transpose(J) )
				x.append( X[0] )
				y.append( k )
				x_err.append( Vx[0][0]**0.5 )
				y_err.append( var_k**0.5 )

	elif param == 'A':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$A$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				I0lg = X[0]
				theta = X[1]
				beta_g = X[2]
				beta_l = X[3]
				A = I0lg**2*beta_g*beta_l
				J = np.array([2*A/I0lg, 0, A/beta_g, A/beta_l])
				var_A = np.matmul( np.matmul( J, Vx ), np.transpose(J) )
				x.append( X[1] )
				y.append( A )
				x_err.append( Vx[1][1]**0.5 )
				y_err.append( var_A**0.5 )

				
	elif param == 'I':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$I$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				I0lg = X[0]
				theta = X[1]
				beta_g = X[2]
				beta_l = X[3]
				z = 1j*beta_l/(np.pi**0.5*beta_g)
				dwdz = -2*z*wofz(z) + 2j/np.pi**0.5
				dzdbg = -z/beta_g
				dzdbl = z/beta_l
				I = I0lg**2*beta_l*wofz(z).real
				J = np.zeros(4)
				J[0] = 2*I/I0lg
				J[2] = beta_l*I0lg**2*( dwdz*dzdbg ).real
				J[3] = I/beta_l + beta_l*I0lg**2*( dwdz*dzdbl ).real
				var_I = np.matmul( np.matmul( J, Vx ), np.transpose(J) )
				x.append( X[1] )
				y.append( I )
				x_err.append( Vx[1][1]**0.5 )
				y_err.append( var_I**0.5 )

	elif param == 'beta':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$I$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				I0lg = X[0]
				theta = X[1]
				beta_g = X[2]
				beta_l = X[3]
				z = 1j*beta_l/(np.pi**0.5*beta_g)
				dwdz = -2*z*wofz(z) + 2j/np.pi**0.5
				dzdbg = -z/beta_g
				dzdbl = z/beta_l
				beta = beta_g / wofz( z ).real
				J = np.zeros(4)
				J[2] = 1 / wofz( z ).real - beta_g*( 1/(wofz( z ))**2 * dzdbg ).real
				J[3] = -beta_g*( 1/(wofz( z )**2) * dzdbl ).real
				var_beta = np.matmul( np.matmul( J, Vx ), np.transpose(J) )
				x.append( X[1] )
				y.append( beta )
				x_err.append( Vx[1][1]**0.5 )
				y_err.append( var_beta**0.5 )

	elif param == 'FWHM':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$FWHM (^o)$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				I0lg = X[0]
				theta = X[1]
				beta_g = X[2]
				beta_l = X[3]
				k = beta_l/(beta_g*np.pi**0.5)
				FWHM = 2*beta_g/np.pi**0.5*(1+k**2)**0.5
				J = np.zeros(4)
				J[2] = 2/np.pi**0.5*(1+k**2)**0.5 - 2*k**2/(np.pi*(1+k**2))**0.5
				J[3] = 2*k/np.pi/(1+k**2)**0.5
				var_FWHM = np.matmul( np.matmul( J, Vx ), np.transpose(J) )
				x.append( X[1] )
				y.append( FWHM )
				x_err.append( Vx[1][1]**0.5 )
				y_err.append( var_FWHM**0.5 )

	elif param == 'form_fact':
		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'Form factor $=\frac{FWHM}{\beta}$ reg.'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				I0lg = X[0]
				theta = X[1]
				beta_g = X[2]
				beta_l = X[3]
				k = beta_l/(beta_g*np.pi**0.5)
				FWHM = 2*beta_g/np.pi**0.5*(1+k**2)**0.5
				z = 1j*beta_l/(np.pi**0.5*beta_g)
				dwdz = -2*z*wofz(z) + 2j/np.pi**0.5
				dzdbg = -z/beta_g
				dzdbl = z/beta_l
				beta = beta_g / wofz( z ).real
				form_fact = FWHM/beta
				dFWHMdbetag = 2/np.pi**0.5*(1+k**2)**0.5 - 2*k**2/(np.pi*(1+k**2))**0.5
				dFWHMdbetal = 2*k/np.pi/(1+k**2)**0.5
				dbetadbetag = 1 / wofz( z ).real - beta_g*( 1/(wofz( z ))**2 * dzdbg ).real
				dbetadbetal = -beta_g*( 1/(wofz( z )**2) * dzdbl ).real
				J = np.zeros(4)
				J[2] = 1/beta*dFWHMdbetag - FWHM/beta**2*dbetadbetag
				J[2] = 1/beta*dFWHMdbetal - FWHM/beta**2*dbetadbetal
				var_form_fact = np.matmul( np.matmul( J, Vx ), np.transpose(J) )
				x.append( X[1] )
				y.append( form_fact )
				x_err.append( Vx[1][1]**0.5 )
				y_err.append( var_form_fact**0.5 )
	elif param == 'A/R':
		if dict_phases == {}:
			print('Dictionnaire des phases requis')
			return

		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'I r\'{e}gression / th\'{e}orique $=\frac{A}{R}$'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				I0lg = X[0]
				theta = X[1]
				beta_g = X[2]
				beta_l = X[3]
				A = I0lg**2*beta_g*beta_l
				J = np.array([2*A/I0lg, 0, A/beta_g, A/beta_l])
				sig_A = ( np.matmul( np.matmul( J, Vx ), np.transpose(J) ) )**0.5

				phase_data = dict_phases[ phase_id ]
				for j in range( len( phase_data[5] ) ):
					if spectre.peak_list[i][5] == phase_data[5][j][0]:
						R, sig_R = phase_data[5][j][8]

				AR = A/R
				sig_AR = A/R*(sig_A**2/A**2 + sig_R**2/R**2)**0.5
				
				x.append( theta )
				y.append( AR )
				x_err.append( Vx[1][1]**0.5 )
				y_err.append( sig_AR )

	elif param == 'delta_th':
		if dict_phases == {}:
			print('Dictionnaire des phases requis')
			return

		x_label = r'$(2 \theta)_{obs} (^o)$'
		y_label = r'$(2 \theta)_{obs} - (2 \theta)_{calc} (^o)'
		for i in range( len( spectre.peak_list ) ):
			if spectre.peak_list[i][3] == True and spectre.peak_list[i][4] == phase_id:
				pic = spectre.peak_list[i][0]
				cle_I0lg = 'p' + str(pic) + '1'
				cle_th = 'p' + str(pic) + '2'
				cle_bg = 'p' + str(pic) + '3'
				cle_bl = 'p' + str(pic) + '4'
				X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
				theta_mes = X[1]
				sig_theta_mes = Vx[1][1]**0.5

				phase_data = dict_phases[ phase_id ]
				for j in range( len( phase_data[5] ) ):
					if spectre.peak_list[i][5] == phase_data[5][j][0]:
						theta_calc = 360./np.pi*phase_data[5][j][3][0] 
						sig_theta_calc = 360./np.pi*phase_data[5][j][3][1]


				delta_theta = theta_mes - theta_calc
				sig_delta_theta = (sig_theta_mes**2 + sig_theta_calc**2)**0.5

				x.append( theta_mes )
				y.append( delta_theta )
				x_err.append( sig_theta_mes )
				y_err.append( sig_delta_theta )

	plt.errorbar( x, y, xerr=x_err, yerr=y_err, fmt = 'o', ecolor = 'g' )
	plt.xlabel( x_label )
	plt.ylabel( y_label )
	if affich == 1:
		plt.show()

	return x, y, x_err, y_err

def plot_aberration( spectre, dict_phases, phase_id, instrum_data ):
	"""

	Trace graphiquement les corrections angulaires dues aux aberrations géométriques superposées avec 
	delta_th, qui correspond à la différence entre les positions mesurée et théorique des pics.
	
	Args :
		spectre
		dict_phases
		phase_id
		instrum_data

	"""

	from scipy.constants import N_A, physical_constants
	r_e = physical_constants['classical electron radius'][0]

	[mat, emetteur, raie] = dict_phases['MatBase']
	[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = read_melange( mat, emetteur, raie )

	mu_l = mu_m * rho

	R_diff = instrum_data.R_diff
	DS = instrum_data.DS
	h = instrum_data.mask / 2.
	delta_sol = instrum_data.delta_sol
	[corr_zero_s, s_s, K1_s, K2_s] = instrum_data.corr_th
	corr_zero = corr_zero_s[0]
	s = s_s[0]
	K1 = K1_s[0]
	K2 = K2_s[0]
	corr_zero = corr_zero*360/2/np.pi

	delta_ref = rho * N_A * 100*r_e * (lam*10**-8)**2 * f1 / ( 2 * np.pi * A ) #Indice de réfraction [gisaxs.com/index.php/Refractive_index]
	q = R_diff*delta_sol/h
	Q1 = (1 - 1./3*q + 3./8*q**2 - 1./10*q**3)/(1 - 1./6*q)
	Q2 = (1./4*q**2 - 3./10*q**3)/(1 - 1./6*q)
	
	x = spectre.raw_data.theta * 2*np.pi/360
	f_transp = -np.sin(x)/(2*mu_l*R_diff) *360/(2*np.pi)
	f_flat = -K1*DS**2/(6. * np.tan( x / 2. ) ) * 360/(2*np.pi)
	f_ax = -K2*(delta_sol**2/12 + h**2/(3*R_diff**2))*1./np.tan( x ) * 360/(2*np.pi)
	f_ax_2 = -h**2/(3*R_diff**2)*(Q1/np.tan( x ) + Q2/np.sin( x )) * 360/(2*np.pi)
	f_refr = -2*delta_ref*np.tan( x/2 ) * 360/(2*np.pi)
	if np.abs(s) > 0:
		f_disp = -2*s/R_diff*np.cos(x/2) * 360/(2*np.pi)
	f_tot = f_transp + f_flat + f_ax + f_refr + corr_zero
	 

	theta_vect = []
	fact_I3p_vect = []

	for i in range( len( spectre.peak_list ) ):
		PSF = spectre.peak_list[i][1][0]
		hkl = spectre.peak_list[i][5]
		if PSF == 'v2' or PSF[0:3] == 'v2k' or PSF[0:3] == 'v3k':
			I0lg = spectre.peak_list[i][1][1]
			theta_mes = spectre.peak_list[i][1][2]
			beta_g = spectre.peak_list[i][1][3]
			beta_l = spectre.peak_list[i][1][4]
			k = beta_l / (np.pi**0.5*beta_g)
			f_I2p = lambda z: -2*beta_l*I0lg**2*np.pi/beta_g**2 * (wofz(z) - 2*z**2*wofz(z) + 2j*z/np.pi**0.5).real
			f_I3p = lambda z: -2*beta_l*I0lg**2*np.pi**1.5/beta_g**3*(-6*z*wofz(z)+4j/np.pi**0.5+4*z**3*wofz(z)-4j*z**2/np.pi**0.5).real
			FWHM = 2*beta_g/np.pi**0.5*(1+k**2)**0.5

		x = theta_mes * 2*np.pi/360
		corr_disp = -2*s/R_diff*np.cos(x/2) * 360/(2*np.pi)
		corr_transp = -np.sin(x)/(2*mu_l*R_diff) *360/(2*np.pi)
		corr_flat = -K1*DS**2/(6. * np.tan( x / 2. ) ) * 360/(2*np.pi)
		corr_ax = -K2*(delta_sol**2/12 + h**2/(3*R_diff**2))*1./np.tan( x ) * 360/(2*np.pi)
		corr_refr = -2*delta_ref*np.tan( x/2 ) * 360/(2*np.pi)

		beta=0
		gamma=0
		RS=1
		x = theta_mes * 2*np.pi/360
		corr_disp = 2*s/R_diff*np.cos(x/2) * 360/(2*np.pi)
		corr_transp = -np.sin(x)/(2*mu_l*R_diff) *360/(2*np.pi)
		corr_flat = -DS**2/(6. * np.tan( x / 2. ) ) * 360/(2*np.pi)
		corr_ax = -(delta_sol**2/12 + h**2/(3*R_diff**2))*1./np.tan( x ) * 360/(2*np.pi)
		corr_refr = -2*delta_ref*np.tan( x/2 ) * 360/(2*np.pi)

		beta=0
		gamma=0
		RS=1
		p = 1e-4
		lam_1 = 1.540562
		lam_2 = 1.544390
		lam_3 = 1.53416
		F_w = 0.4*np.sin(6*2*np.pi/360)
		W_transp = (np.sin(x))**2/(4*R_diff**2*mu_l**2)
		W_mis_sett = beta**2*DS**2/(3*(np.tan(x/2))**2)
		W_incl = 4*gamma**2*h**2*(np.cos(x/2))**2/(3*R_diff**2)
		W_focal = F_w**2/(12*R_diff**2)
		W_RS = RS**2/(12*R_diff**2)
		W_ax = 7*(delta_sol**4/720 + h**4/(45*R_diff**4))*(1/np.tan(x))**2 + h**2/(9*R_diff**2*(np.sin(x))**2)
		W_refr = delta_ref**2/(4*mu_l*p)*(-6*np.log(delta_sol/2) + 25 )
		W_tot = W_transp + W_mis_sett + W_focal + W_RS + W_ax + W_refr 

		if PSF == 'v2':
			fact_I3p = 0
		elif PSF[0:3] == 'v2k':
			th0 = theta_mes
			th2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(th0*np.pi/360) )
			z2 = np.pi**0.5*(th2-th0)/beta_g + 1j*k
			I2p = f_I2p( z2 )
			I3p = f_I3p( z2 )
			fact_I3p = W_tot*I3p/(2*I2p)

		elif PSF[0:3] == 'v3k':
			th0 = theta_mes
			th2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(th0*np.pi/360) )
			z2 = np.pi**0.5*(th2-th0)/beta_g + 1j*k
			th3 = 360/np.pi*np.arcsin( lam_3/lam_1*np.sin(th0*np.pi/360) )
			z3 = np.pi**0.5*(th3-th0)/beta_g + 1j*k
			I2p = f_I2p( z2 ) + f_I2p( z3 )
			I3p = f_I3p( z2 ) + f_I3p( z3 )
			fact_I3p = W_tot*I3p/(2*I2p)
		F_w = 0.4*np.sin(6*2*np.pi/360)
		W_transp = (np.sin(x))**2/(4*R_diff**2*mu_l**2)
		W_mis_sett = beta**2*DS**2/(3*(np.tan(x/2))**2)
		W_incl = 4*gamma**2*h**2*(np.cos(x/2))**2/(3*R_diff**2)
		W_focal = F_w**2/(12*R_diff**2)
		W_RS = RS**2/(12*R_diff**2)
		W_ax = 7*(delta_sol**4/720 + h**4/(45*R_diff**2))*(1/np.tan(x))**2 + h**2/(9*R_diff**2*(np.sin(x))**2)
		W_refr = delta_ref**2/(4*mu_l*p)*(-6*np.log(delta_sol/2) + 25 )
		W_tot = W_transp + W_mis_sett + W_focal + W_RS + W_ax + W_refr 

		if PSF == 'v2':
			fact_I3p = 0
		elif PSF[0:3] == 'v2k':
			th0 = theta_mes
			th2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(th0*np.pi/360) )
			z2 = np.pi**0.5*(th2-th0)/beta_g + 1j*k
			I2p = f_I2p( z2 )
			I3p = f_I3p( z2 )
			fact_I3p = W_tot*I3p/(2*I2p)

		elif PSF[0:3] == 'v3k':
			th0 = theta_mes
			th2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(th0*np.pi/360) )
			z2 = np.pi**0.5*(th2-th0)/beta_g + 1j*k
			th3 = 360/np.pi*np.arcsin( lam_3/lam_1*np.sin(th0*np.pi/360) )
			z3 = np.pi**0.5*(th3-th0)/beta_g + 1j*k
			I2p = f_I2p( z2 ) + f_I2p( z3 )
			I3p = f_I3p( z2 ) + f_I3p( z3 )
			fact_I3p = W_tot*I3p/(2*I2p)

		theta_vect.append( theta_mes )
		fact_I3p_vect.append( fact_I3p )

	param_anal( spectre, phase_id, 'delta_th', dict_phases, affich = 0 )
	#plt.plot( spectre.raw_data.theta, f_transp, 'c', label = 'Correction transparence' )
	plt.plot( spectre.raw_data.theta, f_flat, 'k--', label = r'Correction \'{e}ch. plat' )
	plt.plot( spectre.raw_data.theta, f_ax, 'k-.', label = 'Correction pour divergence axiale' )
	#plt.plot( spectre.raw_data.theta, f_ax_2, 'b:', label = 'Correction pour divergence axiale (2)' )
	#plt.plot( spectre.raw_data.theta, f_refr, 'r', label = r'Correction pour r\'{e}fraction' )
	#plt.plot( theta_vect, fact_I3p_vect, 'r^', label = r'$\frac{W I\prime\prime\prime}{2 I\prime\prime}$' )
	plt.plot( [spectre.raw_data.theta[0], spectre.raw_data.theta[-1]], [corr_zero, corr_zero], 'k:', label = r'Correction z\'{e}ro' )
	if np.abs(s) > 0:
		plt.plot( spectre.raw_data.theta, f_disp, 'y', label = r'D\'{e}placement \'{e}chantillon' )
		f_tot += f_disp
	plt.plot( spectre.raw_data.theta, f_tot, 'k', label = 'Correction totale' )
	plt.legend()
	plt.show()


def calc_aberr( spectre, dict_phases, phase_id, instrum_data ):
	"""

	Calcule et affiche à l'écran les données de correction des pics : position mesurée, position théorique, correcions,
	position corrigée, variance et écart-type de chaque donnée ou paramètre calculé.

	Args :
		spectre
		dict_phases
		phase_id
		instrum_data

	"""
	from scipy.constants import N_A, physical_constants
	r_e = physical_constants['classical electron radius'][0]
	lam_1 = 1.540562
	lam_2 = 1.544390
	lam_3 = 1.53416

	[mat, emetteur, raie] = dict_phases['MatBase']
	[(mu_m, sig_mu_m), (A, sig_A), (rho, sig_rho), (lam, sig_lam), (f1, sig_f1), (f2, sig_f2)] = read_melange( mat, emetteur, raie )

	mu_l = mu_m * rho

	R_diff = instrum_data.R_diff
	DS = instrum_data.DS
	h = instrum_data.mask / 2.
	[corr_zero_s, s_s, K1_s, K2_s] = instrum_data.corr_th
	corr_zero = corr_zero_s[0]
	s = s_s[0]
	K1 = K1_s[0]
	K2 = K2_s[0]
	delta_sol = instrum_data.delta_sol
	delta_ref = rho * N_A * 100*r_e * (lam*10**-8)**2 * f1 / ( 2 * np.pi * A ) #Indice de réfraction [gisaxs.com/index.php/Refractive_index]
	p = 1e-4 #Grosseur des particules, environ 1 micrometre. p en cm

	print( 'Matériau : ' + mat[0][0] )
	print( 'Phase : ' + phase_id )
	print( 'Émetteur ' + emetteur + 'K-' + raie )
	print( 'Longueur d\'onde : %.3f' % lam )
	print( '-----' )
#		(1, 1, 1)  123.4564  12345678  12345678  12345678  12345678  12345678  12345678
	W_vect = []
	for i in range( len( spectre.peak_list ) ):
		PSF = spectre.peak_list[i][1][0]
		hkl = spectre.peak_list[i][5]
		pic = spectre.peak_list[i][0]
		cle_I0lg = 'p' + str(pic) + '1'
		cle_th = 'p' + str(pic) + '2'
		cle_bg = 'p' + str(pic) + '3'
		cle_bl = 'p' + str(pic) + '4'
		X, Vx = constr_Vx( spectre, [cle_I0lg, cle_th, cle_bg, cle_bl] )
		theta_mes = X[1]
		sig_theta_mes = Vx[1][1]**0.5
		I0lg = X[0]
		theta = X[1]
		beta_g = X[2]
		beta_l = X[3]
		k = beta_l / (np.pi**0.5*beta_g)
		f_I2p = lambda z: -2*beta_l*I0lg**2*np.pi/beta_g**2 * (wofz(z) - 2*z**2*wofz(z) + 2j*z/np.pi**0.5).real
		f_I3p = lambda z: -2*beta_l*I0lg**2*np.pi**1.5/beta_g**3*(-6*z*wofz(z)+4j/np.pi**0.5+4*z**3*wofz(z)-4j*z**2/np.pi**0.5).real
		FWHM = 2*beta_g/np.pi**0.5*(1+k**2)**0.5

		x = theta_mes * 2*np.pi/360
		corr_disp = -2*s/R_diff*np.cos(x/2) * 360/(2*np.pi)
		corr_transp = -np.sin(x)/(2*mu_l*R_diff) *360/(2*np.pi)
		corr_flat = -K1*DS**2/(6. * np.tan( x / 2. ) ) * 360/(2*np.pi)
		corr_ax = -K2*(delta_sol**2/12 + h**2/(3*R_diff**2))*1./np.tan( x ) * 360/(2*np.pi)
		corr_refr = -2*delta_ref*np.tan( x/2 ) * 360/(2*np.pi)

		beta=0
		gamma=0
		RS=1
		F_w = 0.4*np.sin(6*2*np.pi/360)
		W_transp = (np.sin(x))**2/(4*R_diff**2*mu_l**2)
		W_mis_sett = beta**2*DS**2/(3*(np.tan(x/2))**2)
		W_incl = 4*gamma**2*h**2*(np.cos(x/2))**2/(3*R_diff**2)
		W_flat = 1./45*DS**4/(np.tan(x/2))**2
		W_focal = F_w**2/(12*R_diff**2)
		W_RS = RS**2/(12*R_diff**2)
		W_ax = 7*(delta_sol**4/720 + h**4/(45*R_diff**4))*(1/np.tan(x))**2 + h**2/(9*R_diff**2*(np.sin(x))**2)
		W_refr = delta_ref**2/(4*mu_l*p)*(-6*np.log(delta_sol/2) + 25 )
		W_tot = W_transp + W_mis_sett + W_focal + W_RS + W_ax + W_refr 

		if PSF == 'v2':
			fact_I3p = 0
		elif PSF[0:3] == 'v2k':
			th0 = theta_mes
			th2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(th0*np.pi/360) )
			z2 = np.pi**0.5*(th2-th0)/beta_g + 1j*k
			I2p = f_I2p( z2 )
			I3p = f_I3p( z2 )
			fact_I3p = W_tot*I3p/(2*I2p)

		elif PSF[0:3] == 'v3k':
			th0 = theta_mes
			th2 = 360/np.pi*np.arcsin( lam_2/lam_1*np.sin(th0*np.pi/360) )
			z2 = np.pi**0.5*(th2-th0)/beta_g + 1j*k
			th3 = 360/np.pi*np.arcsin( lam_3/lam_1*np.sin(th0*np.pi/360) )
			z3 = np.pi**0.5*(th3-th0)/beta_g + 1j*k
			I2p = f_I2p( z2 ) + f_I2p( z3 )
			I3p = f_I3p( z2 ) + f_I3p( z3 )
			fact_I3p = W_tot*I3p/(2*I2p)
	
		W_vect.append([ W_transp, W_mis_sett, W_focal, W_RS, W_ax, W_refr, fact_I3p, FWHM ])	
		corr_tot = corr_zero + corr_disp + corr_transp + corr_flat + corr_ax + corr_refr + fact_I3p
		theta_corr = theta_mes - corr_tot

		phase_data = dict_phases[ phase_id ]
		for j in range( len( phase_data[5] ) ):
			if hkl == phase_data[5][j][0]:
				theta_calc = 360./np.pi*phase_data[5][j][3][0] 
				sig_theta_calc = 360./np.pi*phase_data[5][j][3][1]

		var_pic_corr = W_tot + sig_theta_mes**2
		sig_theta_corr = var_pic_corr**0.5

		delta_theta = theta_mes - theta_calc
		sig_delta_theta = (sig_theta_mes**2 + sig_theta_calc**2)**0.5

		delta_theta_corr = theta_corr - theta_calc
		sig_delta_theta_corr = (sig_theta_corr**2 + sig_theta_calc**2)**0.5

		print( 'Pic : ' + str(hkl) )
		print( '			(2_th) (deg)  Var (rad^2)  sig (rad)  sig(deg)  FWHM (deg)')	
		#				123.2345      1.2345e-02   1.2345     1.2345    1.2345

		print( 'Pos. centroide (mes)    %8.4f      %9.4e   %.4f     %.4f    %.4f' %(theta_mes, (sig_theta_mes*2*np.pi/360)**2, sig_theta_mes*2*np.pi/360, sig_theta_mes, 2*sig_theta_mes) )
		print( 'Pos. pic (mes)		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(theta_mes, (FWHM*np.pi/360)**2, FWHM*np.pi/360, FWHM/2, FWHM ) )
		print( 'Pos. pic (calc)		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(theta_calc, (sig_theta_calc*2*np.pi/360)**2, sig_theta_calc*2*np.pi/360, sig_theta_calc, 2*sig_theta_calc) )
		print( 'MES - CALC		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(delta_theta, (sig_delta_theta*2*np.pi/360)**2, sig_delta_theta*2*np.pi/360, sig_delta_theta, 2*sig_delta_theta) )
		print( '\n' )
		print(u'Correction zéro		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(corr_zero, 0, 0, 0, 0) )
		print(u'Correction déplacement	%8.4f      %9.4e   %.4f     %.4f    %.4f' %(corr_disp, 0, 0, 0, 0) )
		print(u'Correction éch. plat	%8.4f      %9.4e   %.4f     %.4f    %.4f' %(corr_flat, W_flat, W_flat**0.5, W_flat**0.5*360/(2*np.pi), W_flat**0.5*360/np.pi) )
		print( 'Correction div. axiale	%8.4f      %9.4e   %.4f     %.4f    %.4f' %(corr_ax, W_ax, W_ax**0.5, W_ax**0.5*360/(2*np.pi), W_ax**0.5*360/np.pi) )
		print( 'TOTAL			%8.4f      %9.4e   %.4f     %.4f    %.4f' %(corr_tot-fact_I3p, W_tot, W_tot**0.5, W_tot**0.5*360/(2*np.pi), W_tot**0.5*360/np.pi) )
		
		print( '\nPic corrigé		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(theta_corr, var_pic_corr, var_pic_corr**0.5, var_pic_corr**0.5*360/(2*np.pi), var_pic_corr**0.5*360/np.pi) )

		print( 'Pos. pic (calc)		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(theta_calc, (sig_theta_calc*2*np.pi/360)**2, sig_theta_calc*2*np.pi/360, sig_theta_calc, 2*sig_theta_calc) )
		print( 'CORR - CALC		%8.4f      %9.4e   %.4f     %.4f    %.4f' %(delta_theta_corr, (sig_delta_theta_corr*2*np.pi/360)**2, sig_delta_theta_corr*2*np.pi/360, sig_delta_theta_corr, 2*sig_delta_theta_corr) )
		print( '\nI\'\' : %8.4e; I\'\'\' : %8.4e; WI\'\'\'/2I\'\' : %.4f' %(I2p, I3p, fact_I3p) )

		
def opt_aberr( spectre, dict_phases, phase_id, instrum_data ):
	"""
	
	Utilise la méthode des moindres carrés non-linéaire restreints pour trouver la meilleure fonction de correction
	de la position des pics.

	Paramètres :
		dth0 : Correction du zéro du diffractomètre : dans [-1e-3, 1e-3] rad
		s : Déplacement de l'échantillon par rapport au point focal du diffractomètre : dans [-0.5, 0.5] mm
		K1 : Facteur d'échelle de la correction pour échantillon plat : dans [0, 10]
		K2 : Facteur d'échelle de la correction pour divergence axiale : dans [0, 10]

	Args :
		spectre
		dict_phases
		phase_id
		instrum_data

	Extrants :
		beta : Vecteur contenant les paramètres optimisés : [dth0, s, K1, K2]
		V_beta : Matrice de variance-covariance des paramètres optimisés

	"""
	
	A = param_anal( spectre, phase_id, 'delta_th', dict_phases, affich=0 )
	xdata = np.asarray(A[0])*np.pi/360 #x = theta (rad)
	ydata = np.asarray(A[1])*2*np.pi/360 #y = 2_theta(obs) - 2_theta(theo) (rad)
	sigma = (np.asarray(A[3])*2*np.pi/360) #rad 
	h = instrum_data.mask/2.
	R = instrum_data.R_diff
	DS = instrum_data.DS
	d_sol = instrum_data.delta_sol
	n = len( spectre.peak_list )

	f = lambda th, dth0, s, K1, K2: dth0 - 2.*s*np.cos(th)/R - K1*DS**2/(6*np.tan(th)) - K2*(d_sol**2/12 + h**2/(3*R**2))/np.tan(2*th)
	beta_0 = [0, 0, 1, 1]
	bounds = ([-1e-3, -0.5, 0, 0], [1e-3, 0.5, 10, 10])
	#th : 2_theta
	beta, V_beta = curve_fit( f, xdata, ydata, p0 = beta_0, sigma = sigma, absolute_sigma = True, bounds = bounds )

	dth0 = beta[0]
	sig_dth0 = V_beta[0, 0]**0.5
	s = beta[1]
	sig_s = V_beta[1, 1]**0.5
	K1 = beta[2]
	sig_K1 = V_beta[2, 2]**0.5
	K2 = beta[3]
	sig_K2 = V_beta[3, 3]**0.5
	f = lambda th: dth0 - 2.*s*np.cos(th)/R - K1*DS**2/(6*np.tan(th)) - K2*(d_sol**2/12 + h**2/(3*R**2))/np.tan(2*th)

	n = len(xdata)
	m = 4
	chi2 = 0
	for i in range( n ):
		yi = ydata[i]
		fi = f(xdata[i])
		si = sigma[i]
		chi2 += (yi - fi)**2/si**2
	GOF = (chi2 / (n-m))**0.5	

#	J = np.zeros( (n, 4) )
#	for i in range( n ):
#		theta = spectre.peak_list[i][1][2]*np.pi/360.
#		J[i, 0] = 1
#		J[i, 1] = -2*np.cos(theta)/instrum_data.R_diff
#		J[i, 2] = -instrum_data.DS**2/(6*np.tan(theta))
#		J[i, 3] = -(instrum_data.delta_sol**2/12 + h**2/(3*instrum_data.R_diff**2))/np.tan(2*theta)
#
#	print J
#	JTWJ = np.matmul( np.transpose(J), np.matmul( W, J ) )
#	JTWy = np.matmul( np.transpose(J), np.matmul( W, y ) )
#	beta = np.matmul( np.linalg.inv(JTWJ), JTWy )
#	V_beta = np.linalg.inv( JTWJ )

	instrum_data.corr_th = [(dth0, sig_dth0), (s, sig_s), (K1, sig_K1), (K2, sig_K2) ]
	return instrum_data, beta, V_beta, GOF


def opt_FWHM( spectre, dict_phases, phase_id, instrum_data, affich = 0 ):

	A = param_anal( spectre, phase_id, 'FWHM', dict_phases, affich=0 )
	xdata = np.asarray(A[0])*np.pi/360 #x = theta (rad)
	ydata = np.asarray(A[1])*2*np.pi/360 #y = FWHM (rad)
	sigma = np.asarray(A[3])*2*np.pi/360 #rad 

	f = lambda th, U, V, W: (U*(np.tan(th))**2 + V*np.tan(th) + W)**0.5 
	beta, V_beta = curve_fit( f, xdata, ydata, sigma = sigma, bounds = ([0, 0, 0], [10, 10, 10]), absolute_sigma = True )

	U = beta[0]
	sig_U = V_beta[0, 0]**0.5
	V = beta[1]
	sig_V = V_beta[1, 1]**0.5
	W = beta[2]
	sig_W = V_beta[2, 2]**0.5
	f = lambda th: (U*(np.tan(th))**2 + V*np.tan(th) + W)**0.5 * 360/(2*np.pi) #FWHM(o)

	n = len(xdata)
	m = 3
	chi2 = 0
	for i in range( n ):
		yi = ydata[i]
		fi = f(xdata[i])*2*np.pi/360
		si = sigma[i]
		chi2 += (yi - fi)**2/si**2
	GOF = (chi2 / (n-m))**0.5	


	if affich == 1 :
		print( 'U : %s rad^2' %( write_nmbr( U, V_beta[0, 0]**0.5 ) ) )
		print( 'V : %s rad^2' %( write_nmbr( V, V_beta[1, 1]**0.5 ) ) )
		print( 'W : %s rad^2' %( write_nmbr( W, V_beta[2, 2]**0.5 ) ) )
		print( 'GOF : %.4f' %(GOF) )
		plt.figure()
		plt.errorbar( xdata*360/np.pi, ydata*360/(2*np.pi), yerr = sigma*360/(2*np.pi), fmt = 'o', ecolor = 'g' )
		plt.plot( spectre.raw_data.theta, f( spectre.raw_data.theta*np.pi/360 ), 'k-' )
		plt.xlabel( r'$(2 \theta)_{obs} (^o)$' )
		plt.ylabel( r'$FWHM (^o)$ reg.' )
		plt.show()
	
	instrum_data.calib_FWHM = [ (U, sig_U), (V, sig_V), (W, sig_W) ]

	return instrum_data, beta, V_beta, GOF
		
	
def opt_form_fact( spectre, dict_phases, phase_id, instrum_data, affich = 0 ):

	A = param_anal( spectre, phase_id, 'form_fact', dict_phases, affich=0 )
	xdata = np.asarray(A[0]) #x = 2*theta (o)
	ydata = np.asarray(A[1]) #y = form_fact
	sigma = np.asarray(A[3]) #(o) 

	f = lambda th, b, c: 0*th**2 + b*th + c 
	beta, V_beta = curve_fit( f, xdata, ydata, sigma = sigma, absolute_sigma = True )
	b = beta[0]
	sig_b = V_beta[0, 0]**0.5
	c = beta[1]
	sig_c = V_beta[1, 1]**0.5
	f = lambda th:  b*th + c 

	n = len(xdata)
	m = 2
	chi2 = 0
	for i in range( n ):
		yi = ydata[i]
		fi = f(xdata[i])
		si = sigma[i]
		chi2 += (yi - fi)**2/si**2
	GOF = (chi2 / (n-m))**0.5	


	if affich == 1 :
		#print( 'a : %s deg^-2' %( write_nmbr( a, V_beta[0, 0]**0.5 ) ) )
		print( 'b : %s deg^-1' %( write_nmbr( b, V_beta[0, 0]**0.5 ) ) )
		print( 'c : %s ' %( write_nmbr( c, V_beta[1, 1]**0.5 ) ) )
		print( 'GOF : %.4f' %(GOF) )
		plt.figure()
		plt.errorbar( xdata, ydata, yerr = sigma, fmt = 'o', ecolor = 'g' )
		plt.plot( spectre.raw_data.theta, f( spectre.raw_data.theta ), 'k-' )
		plt.xlabel( r'$(2 \theta)_{obs} (^o)$' )
		plt.ylabel( r'Form factor $=\frac{FWHM}{\beta}$ reg.' )
		plt.show()
	
	instrum_data.calib_ff = [ (b, sig_b), (c, sig_c) ]

	return instrum_data, beta, V_beta, GOF

def read_nmbr( nmbr ):
	"""
	
	Lit une chaîne de caractère contenant le nombre et l'écart-type et le transforme en tuple contenant ces deux données.

	Intrants :
		nmbr : Chaîne de caractères de format '1.234(56)e+78'

	Extrants :
		(nombre, incertitude)

	"""

	def isint( car ):
		try:
			test = int(car)
			return True
		except ValueError:
			return False
	
	chiffre = False
	virgule = False
	exposant = 0
	nombre_str = ''
	is_incert = False
	is_exposant = False
	is_zero = False
	exp_str = ''
	incert_str = ''
	negatif = False

	for i in range( len(nmbr) ):
		car = nmbr[i]
		if car == '-' or car == '.' or car == '(' or car == ')' or car == 'e' or car == '+':
			pass
		else:
			if not isint( car ):
				print('Erreur de format')
				return

		if i == 0 and car == '-':
			negatif = True
			continue
		
		elif chiffre == False and virgule == False and is_zero == False:
			if car == '0':
				pass
			elif car == '.':
				exposant -= 1
				virgule = True
			elif isint( car ):
				nombre_str += car + '.'
				chiffre = True
			elif car == 'e' or car == 'E':
				is_exposant = True
				is_zero = True
			elif car == '(':
				is_incert = True
				is_zero = True

			else:
				print('Erreur de format')
				return

		elif chiffre == False and virgule == True and is_zero == False:	
			if car == '0':
				exposant -= 1
			elif isint( car ):
				chiffre = True
				nombre_str += car + '.'
			elif car == 'e' or car == 'E':
				is_exposant = True
				is_zero = True
			elif car == '(':
				is_incert = True
				is_zero = True
			else:
				print('Erreur de format')
				return

		elif is_exposant == True:
			if isint( car ) or car == '-':
				exp_str += car
				
		elif is_incert == True:
			if isint( car ):
				incert_str += car
			elif car == ')':
				is_incert = False

		elif (chiffre == True and virgule == False) or is_zero == True:
			if car == '.':
				virgule = True
			elif isint( car ):
				nombre_str += car
				exposant += 1
			elif car == '(':
				is_incert = True
			elif car == 'e' or car == 'E':
				is_exposant = True
			else:
				print('Erreur de format')
				return
		
		elif chiffre == True and virgule == True:
			if isint( car ):
				nombre_str += car
			elif car == '(':
				is_incert = True
			elif car == 'e' or car == 'E':
				is_exposant = True
			else:
				print('Erreur de format')
				return

	if exp_str == '':
		pass
	else:
		exposant += int( exp_str )
	
	if nombre_str == '' or is_zero == True or float( nombre_str ) == 0.0:
		nombre_str = '0.'
		exposant = -2
	
	nombre = float( nombre_str )*10**exposant 

	exp_incert = exposant - len( nombre_str ) + 2
	
	if incert_str == '':
		incert = 0
	else:
		incert = float(incert_str)*10**exp_incert
	
	if negatif == True:
		nombre = -nombre

	return( nombre, incert )
	
def write_nmbr( nombre, incert = 0, ndec = 8 ):
	"""

	Écrit une chaîne de caractère contenant le nombre et l'écart-type à partir d'un tuple contenant ces deux données.

	Intrants :
		nombre 
		incertitude (0 par défaut, si c'est le cas, la variable est considérée comme sans incertitude)
		ndec = 8 : nombre de décimales à afficher dans le cas où il n'y a pas d'incertitude.

	Extrants :
		Chaîne de caractères de format '1.234(56)e+78'

	"""


	if type(nombre) == tuple and len(nombre) == 2:
		incert = nombre[1]
		nombre = nombre[0]
	if incert == 0:
		return '{:.{}e}'.format( nombre, ndec )

	signe = ''
	if nombre < 0:
		signe = '-'
		nombre = -nombre
	expn = np.floor( np.log10( nombre ) )
	expi = np.floor( np.log10( incert ) )
	nsig = int(expn - expi + 1)
	nbr_str = '{:.{}e}'.format( nombre, max(nsig,0) )
	incert_str = '{:.{}e}'.format( incert, 1 )
	incert_parent = incert_str[0] + incert_str[2]
	if nsig == 0:
		return signe + nbr_str[0] + '(' + incert_parent + ')' + nbr_str[1:]
	elif nsig < 0:
		nzeros = 0 - nsig
		for i in range(nzeros):
			incert_parent = incert_parent + '0'

		return signe + nbr_str[0] + '(' + incert_parent + ')' + nbr_str[1:]
	else:
		return signe + nbr_str[:(nsig+2)] + '(' + incert_parent + ')' + nbr_str[(nsig+2):]
