# -*- coding: latin-1 -*-

#######################################################################
# Projet            : DPHM601 Fragilisation par l'hydrog�ne 
#
# Programme         : xrdsim.py
#
# Auteur            : Simon Lalibert�-Riverin
#
# Date de cr�ation  : 20161114
#
# Objectif          : R�gression et d�convolution de spectres de diffraction des rayons X 
#
# Historique r�v.   :
#
# R�v. 	Date    	Auteur      	Description  
# 1	20161114	S. L.-Riverin	1ere version, fonction read seulement
# 2	20161222	S. L.-Riverin	Rajout� read_el, read_melange et mat_conv
# 3	20170112	S. L.-Riverin	Rajout� classes spectre et data
#					Rajout� algorithmes diff�rentiation et lissage de spectre
#					Rajout� find_peak et fonctions associ�es
# 4	20170117	S. L.-Riverin	Rajout� calcul de bruit de fond et de premi�re approximation par Gaussienne
#					Rajout� calcul des diff�rents R
# 5	20170131	S. L.-Riverin	Rajout� fonction iter
#					Impl�ment� PSF lorentzienne
# 6	20170202	S. L.-Riverin	Impl�ment� PSF de Voigt
# 7	20170203	S. L.-Riverin	Modifi� structure de peak_list
#					Modifi� find_peak et trace_peak pour tenir compte de nouvelle struct
#					Rajout� fonctions add_peak et del_peak dans spectre
#					Modifi� fonction fit_approx, fit_calcul et iter pour tenir compte de la nouvelle struct
#					Rajout� fonctions toggle_peak et split_peak
#					Am�lior� r�f�ren�age des coefficients de la matrice jacobienne avec un dictionnaire
#					Reste les fonctions de background � mod�liser
# 8	20170207	S. L.-Riverin	Enlev� la possibilit� de faire une approximation bruit de fond 0d
#					Rajout� la fonction write et adapt� les autres routines pour permettre le fonctionnement
# 9	20170208	S. L.-Riverin	Ajout� PSF 'v2'
# 10	20170216	S. L.-Riverin	Ajout� le mode 4 � fit_approx, pour contr�ler FWHM
# 11	20170328	S. L.-Riverin	Modifi� tag_peaks pour uniformiser format tuple avec xrdanal
#					Modifi� fonction write pour que la sauvegarde des donn�es de convergence soient optionnelles
# 12	20170330	S. L.-Riverin	Rajout� fonction pour est_noise et rajout� invocation de cette fonction dans fit_calcul
# 13	20170406	S. L.-Riverin	Rajout� fonctionnalit�s pour mod�liser la s�paration entre les pics K-alpha1 et Kalpha2. 
#					Fonctions fit_approx, fit_calcul et iter modifi�es. Fonction split_Kalpha rajout�e.
#
#					VERSIONS SUBS�QUENTES : DANS GITHUB
#
#######################################################################


import csv
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import os.path
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.linalg import solve_triangular, solve
from scipy.special import wofz
import copy
import xrdanal
import mater

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.ioff()

class data:
	def __init__( self ):
		self.theta = np.array( [] )	#2-theta
		self.count = np.array( [] )	#Nombre de comptes

class etat_cl:
	def __init__( self ):
		self.raw_data = False
		self.peak_list = False
		self.back_list = False
		self.back = False
		self.reg = False
		self.calcul = False

class spectre:

	def __init__( self ):
		class fit_donnees:
			pass
		self.raw_data = data()
		self.etat = etat_cl()	
		self.back_pts_list = []
		self.peak_list = []
		self.fit = fit_donnees()
		self.fit.delta_vec = np.array([])
		self.fit.R_vec = np.array([])
		self.etat = etat_cl()

	def read( self, filename = '', plot = 0, write = 0 ):
		"""
		Lit un spectre � partir d'un fichier
		Format csv avec formatage particulier :
		
			*****
				<Ent�te, nombre ind�fini de lignes>
				...
				Angle,Intensity
				<data>
				...
			******

		Args :
			filename :	Chemin d'acc�s du fichier � lire
			plot :		plot = 1 affiche le spectre
			write :		write = 1 �crit le contenu dans un fichier .PRN

		"""	
		
		if len( self.raw_data.theta ) != 0:
			r = raw_input( u'�craser donn�es? 1 = oui autre = non : '.encode('cp850') ) 
			if r.isdigit() == True and int( r ) != 1:
				return

		if filename == '':
			filename = raw_input( 'Nom du fichier? : ')
			if not os.path.exists( filename ):
				print( 'Fichier introuvable' )
				return

		self.etat = etat_cl()
		self.etat.raw_data = True	
		self.raw_data = data()
		self.filename = filename
		
		dataread = 0
		with open(filename, 'rb') as csvfile:
			csvreader = csv.reader( csvfile, delimiter = ',' )
			for row in csvreader:
				if dataread == 1:	
					self.raw_data.theta = np.append( self.raw_data.theta, float( row[0] ) )
					self.raw_data.count = np.append( self.raw_data.count, float( row[1] ) )

				if row[0] == 'Angle':
					dataread = 1
					
		if plot == 1:
			plt.plot( self.raw_data.theta, self.raw_data.count )
			plt.show()

		if write == 1:
			f = open( filename + '.PRN', 'w' )
			for i in range( len(theta) ):
				f.write( str( self.raw_data.theta[i] ) + '\t' + str( self.raw_data.count[i] ) + '\n' )
			f.close()	

	def write( self, file_out, trace_iter = False ):
		"""

		�crit un programme python contenant les donn�es. Pour recharger les donn�es, il suffit d'importer le fichier et d'ex�cuter
		la commande read_data(), qui retourne les donn�es dans une structure <spectre>

		Args :	file_out :	Fichier dans lequel enregistrer les donn�es. Attention si le fichier est d�plac�, la r�f�rence
					vers les donn�es brutes (spectre.filename) risque d'�tre bris�e

		"""


		with open( file_out + '.py', 'w' ) as f:
			f.write('import xrdsim\n')
			f.write('import numpy as np\n')
			f.write('def read_data():\n')
			f.write('\tspectre=xrdsim.spectre()\n')
			f.write('\tspectre.read(\'' + self.filename + '\')\n')
			if self.etat.peak_list == True:
				f.write('\tspectre.etat.peak_list = True\n')
				f.write('\tspectre.etat.reg = ' + str(self.etat.reg) + '\n')
				for i in range( len( self.peak_list ) ):
					f.write('\tspectre.peak_list.append([')
					f.write(str( self.peak_list[i][0]) + ', [' )
					for j in range( len( self.peak_list[i][1] ) ):
						if j == 0:
							f.write('\'')
						f.write( str( self.peak_list[i][1][j] ))
						if j == 0:
							f.write('\'')
						f.write(', ')	
					
					f.write('], ' + str( self.peak_list[i][2] ) )
					f.write(', ' + str( self.peak_list[i][3] ) )
					f.write(', \'' + str( self.peak_list[i][4] ) + '\'' )
					f.write(', ' + str( self.peak_list[i][5] ) + '])\n' )

			if self.etat.back_list == True:
				f.write('\tspectre.etat.back_list = True\n')
				
				for i in range( len( self.back_pts_list ) ):
					f.write('\tspectre.back_pts_list.append([')
					f.write( str( self.back_pts_list[i][0] ) + ', ' )
					f.write( str( self.back_pts_list[i][1] ) + '])\n' )

			if self.etat.calcul == True:
				f.write('\tspectre.etat.calcul = True\n')
				if trace_iter == True:
					for i in range( len( self.fit.R_vec ) ):
						f.write('\tspectre.fit.R_vec = np.append(spectre.fit.R_vec, ' + str(self.fit.R_vec[i]) + ')\n')
						f.write('\tspectre.fit.delta_vec = np.append(spectre.fit.delta_vec, ' + str(self.fit.delta_vec[i]) + ')\n')

			f.write('\tspectre.background_approx()\n')
			f.write('\tspectre.fit_calcul()\n')
			f.write('\treturn spectre\n')


	def find_peaks( self, plot = 0, lbl = '', pct_min_d2 = 0.01, tag = 1 ):
		"""
		Trouve les pics � partir des donn�es brutes
		Trouve les z�ros de la d�riv�e apr�s plusieurs lissages
		Discrimine les faux pics avec un crit�re sur la d�riv�e seconde

		Args:
			plot :
				0 : Aucun graphe
				1 : Trace imm�diatement
				2 : Garde en m�moire
				3 : Mode analyse : trace chacune des courbes de d�riv�es et de lissages utilis�es

			tag :
				Mettre � 1 pour identifier les pics

			lbl :	
				�tiquette de donn�es � donner � la s�rie si plot = 2

			pct_min_d2 : 
				Pourcentage de la valeur de la d�riv�e seconde maximale utilis� comme seuil pour discriminer les faux pics
		
		Lorsque le pic est trouv�, il est rajout� � self.peak_list avec la fonction self.add_peak

		"""

		def signe( x ):
			if x >= 0.:
				return 1
			else:
				return 0
		
		if self.etat.raw_data == False:
			print( u'Pas de donn�es brutes' )
			return

		data_smooth = lisser_moy( self.raw_data, n = 3 )
		data_0 = copy.deepcopy( data_smooth )
		data_0.count = data_0.count**0.5
		d_data = diff_data( data_0 )
		d_data_smooth = lisser_moy( d_data, n = 10 )
		d2_data = diff_data( d_data_smooth )
		d2_data_smooth = lisser_moy( d2_data, n = 10 )
		seuil = min( d2_data_smooth.count ) * pct_min_d2

		if plot == 3:
			trace( self.raw_data, plot = 1 )
			trace( data_smooth, plot = 1 )
			trace( data_0, plot = 1 )
			trace( d_data, plot = 1 )
			trace( d_data_smooth, plot = 1 )
			trace( d2_data, plot = 1 )
			plt.figure()
			trace( d2_data_smooth, plot = 2 )
			plt.plot( d2_data_smooth.theta, np.ones( len(d2_data_smooth.theta) ) * seuil )
			plt.show()
		
		self.peak_list=[]
		f_count = interp1d( data_smooth.theta, data_smooth.count )	#L'intensit� des pics trouv�s est calcul�e � partir de la fonction liss�e
		f_d2 = interp1d( d2_data_smooth.theta, d2_data_smooth.count, bounds_error = False, fill_value = 0 )
		
		s = signe( d_data_smooth.count[0] )
		for i in range( len( d_data_smooth.theta ) - 1 ):
			s2 = signe( d_data_smooth.count[i + 1] )
			if s2 != s and f_d2( d_data_smooth.theta[i] ) < seuil:	
				#Rep�re tous les z�ros de la d�riv�e en utilisant le changement de signe comme crit�re
				#Applique aussi le seuil de discrimation des faux pics
				
				theta_peak = 0.5*(d_data_smooth.theta[i] + d_data_smooth.theta[i-1])
				count_peak = f_count( theta_peak )
				
				existe = False
				for j in range( len( self.peak_list ) ):
					if np.abs( self.peak_list[j][2] - theta_peak ) < 0.01:
						existe = True

				if existe == False :
					self.add_peak( theta_peak )
				existe = True	
			
			s = s2
		
		self.data_smooth = data_smooth
		self.etat.peak_list = True

		if tag == 1:
			self.tag_peaks()
			return

		if plot == 1:
			self.trace_peak()
			
	def tag_peaks( self ):
						
		plt.ion()
		self.trace_peak()
		print( u'Pic #\ttheta\tPhase\tPlan' )
		liste_del = []
		for i in range( len( self.peak_list ) ):
			print( str(self.peak_list[i][0]) + '\t' + str(np.round( self.peak_list[i][2])) + '\t' + self.peak_list[i][4] + '\t' + str(self.peak_list[i][5]) )
			instruction = raw_input( '<d> pour effacer, <m> pour modifier, autre touche pour conserver tel quel ... : ' )
			if instruction == 'd':
				liste_del.append( self.peak_list[i][0] )
				
			elif instruction == 'm':
				self.peak_list[i][4] = raw_input( 'Phase? ... : ')
				print('Plan? ... : ')
				h = raw_input( 'h ... : ' )
				k = raw_input( 'k ... : ' )
				l = raw_input( 'l ... : ' )
				
				self.peak_list[i][5] = ( int(h), int(k), int(l) ) 

		for i in range( len( liste_del ) ):
			self.del_peak( liste_del[i] )

	def add_peak( self, theta, plot = 0, phase = '', plan = () ):
		"""
		Rajoute un pic dans la liste de pics self.peak_list.
		
		Args :
			theta : Position du pic � rajouter
			plot : Mettre � 1 pour afficher le r�sultat avec la fonction self.trace_peak
			phase, plan : donn�es cristallographiques

		self.peak_list est une liste dont les �l�ments sont des listes de la forme : [no_seq, reg=[], theta, actif=True, phase, plan] o� :
			no_seq : num�ro unique attribu� � un pic
			reg : structure de liste vide pour le moment qui servira � stocker les donn�es de la r�gression
			theta : Position initiale du pic tel qu'ins�r� (peut raffin� par la r�gression)
			actif : Si mis � <False>, la fonction <iter> va exclure ce pic de l'optimisation. <False> par d�faut s'il n'y a pas d'approximation
			phase, plan : donn�es cristallographiques
		"""
		
		if self.etat.peak_list == False:

			self.peak_list = []
			self.data_smooth = lisser_moy( self.raw_data, n = 3 )
			self.etat.peak_list = True

		#Rajoute un pic avec un num�ro s�quentiel sup�rieur au num�ro max.
		seq_no = -1
		for i in range( len( self.peak_list ) ):
			seq_no = max( seq_no, self.peak_list[i][0] )
		seq_no += 1

		self.peak_list.append( [seq_no, [], theta, False, phase, plan] )

		if plot == 1:
			self.trace_peak()

		return seq_no	

	def del_peak( self, no_seq, plot = 0):
		"""
		Efface un pic � partir de son num�ro s�quentiel
		
		Args :
			no_seq : Num�ro s�quentiel (et non index i dans peak_list[i])
			plot : Mettre � un pour voir le trac� du spectre apr�s l'effacement du pic

		La fonction modifie self.peak_list

		"""

		if self.etat.peak_list == False:
			print( u'Aucun pic � effacer' )
			return

		for i in range( len( self.peak_list ) ):
			if self.peak_list[i][0] == no_seq:
				del self.peak_list[i]
				if plot == 1:
					self.trace_peak()
				return
		
		print( u'Aucun pic de ce num�ro' )

	def toggle_peak( self, liste_no_seq ):
		"""
		Change l'�tat (actif/inactif) d'un pic. Un pic inactif sera int�gr� au calcul par la fonction fit_calcul, mais pas
		mis � jour par l'ex�cution de la fonction iter.

		Args :
			liste_no_seq : liste contenant les pics pour lesquels on veut changer d'�tat.

		La fonction modifie self.peak_list
		"""

		if self.etat.peak_list == False:
			print( 'Aucune liste de pics' )

		for i in range(	len( liste_no_seq ) ):
			for j in range( len( self.peak_list ) ):
				if self.peak_list[j][0] == liste_no_seq[i]:
					if len( self.peak_list[j][1] ) == 0:
						self.peak_list[j][3] = False
					else:
						self.peak_list[j][3] = not self.peak_list[j][3]

#	def split_peak( self, no_seq, ratio, delta_theta, plot = 0 ):
#		if self.etat.peak_list == False:
#			print( 'Aucune liste de pics' )
#			return
#		
#		for i in range( len( self.peak_list ) ):
#			if self.peak_list[i][0] == no_seq:
#				pic_index = i
#
#		if len( self.peak_list[pic_index][1] ) == 0:
#			print( u'Faire une approximation avant de s�parer le pic' )
#			return
#
#		PSF = self.peak_list[pic_index][1][0]
#		if PSF == 'g':
#			th_init = self.peak_list[pic_index][2] 	
#			th0 = self.peak_list[pic_index][1][2]
#			th1 = th0 + delta_theta
#			th2 = th0 - ratio**2/(1 - ratio**2) * delta_theta
#			I0 = self.peak_list[pic_index][1][1]
#			I1 = ratio*I0
#			I2 = (1 - ratio**2)**0.5*I0
#			c0 = self.peak_list[pic_index][1][3]
#			c1 = ratio*c0
#			c2 = (1 - ratio**2)**0.5*c0
#
#			no_seq_nouv = self.add_peak(th1)
#			self.peak_list[len(self.peak_list)-1][1] = ['g', I1, th1, c1]
#			self.peak_list[pic_index][1][1] = I2
#			self.peak_list[pic_index][1][2] = th2
#			self.peak_list[pic_index][1][3] = c2
#			return no_seq_nouv
			

	

	def trace_peak( self, plot = 1, tag = 1 ):
		"""
		Permet de visualiser les donn�es brutes, les donn�es liss�es et la position des pics enregistr�s

		Args :
			plot :
				plot = 1 : Affiche imm�diatement
				plot = 2 : Garde en m�moire
			tag :
				tag = 1 : Affiche le num�ro du pic
				tag = 2 : Affiche la phase et le plan
		
		"""
		
		if self.etat.peak_list == False:
			print( 'Aucune liste de pics' )
			return

		if plot == 1 or plot ==2:
			plt.plot( self.raw_data.theta, self.raw_data.count, label = r'Donn\'{e}es brutes' )
			if self.etat.calcul == False:
				plt.plot( self.data_smooth.theta, self.data_smooth.count, label = r'Donn\'{e}es liss\'{e}es, n = 3' )
				f_count = interp1d( self.data_smooth.theta, self.data_smooth.count )
			else:
				plt.plot( self.fit.data_reg.theta, self.fit.data_reg.count, label = r'Donn\'{e}es de r\'{e}gression' )
				f_count = interp1d( self.fit.data_reg.theta, self.fit.data_reg.count )

			for i in range(len( self.peak_list )):
				theta_peak = self.peak_list[i][2]
				plt.plot( [theta_peak, theta_peak], [0, f_count(theta_peak) ], color='r' )
				if tag == 1:
					plt.annotate( r'\large{' + str(self.peak_list[i][0]) + '}' , (theta_peak, f_count(theta_peak) ) )
				elif tag == 2 :	
					plt.annotate( r'\large{$' + self.peak_list[i][4] + '_{(' + self.peak_list[i][5] + ')}$}', (theta_peak, f_count(theta_peak)))

				
			#plt.plot( self.peak_list.theta, self.peak_list.count, 'ro' ) r7
			
			if plot == 1:
				plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
				plt.ylabel( r'$I$ (comptes)' )
				plt.legend()
				plt.show()

	

	def fit_approx( self, plot = 0, mode = 1, liste_pics = [], PSF = 'v2', FWHM_appr = 0., I_appr = 0. ):
		"""
		� partir de la liste des pics enregistr�s, fait une premi�re approximation des pics avec des fonctions de r�g.

		Args :
			mode :
				1 : Approxime seulement les pics pour lesquels aucune r�gression n'existe
				2 : Approxime les pics dans liste_pics
				3 : Approxime tous les pics et �crase les r�gressions pr�c�dentes
				4 : Approxime un pic avec des valeurs de I et de FWHM entr�es manuellement
			liste_pics : Liste des pics � modifier. Utile seulement si mode = 2. [no_seq1, no_seq2, ... ] 	
			PSF :	Type de fonction de profil
			FWHM_appr, I_appr : param�tres entr�s manuellement pour le mode 4.
		
		D�finition de la Gaussienne (PSF = 'g') : f(x) = a*exp( -(x-b)^2 / (2c^2) )
			a :	Intensit�. Utilise la valeur enregistr�e dans la liste de pics. Le bruit est soustrait s'il est mod�lis�.
			b :	Position du pic. Utilise la valeur enregistr�e dans la liste de pics.
			c :	�cart-type de la distribution = FWHM / ( 2*sqrt(2*ln2) )

		D�finition de la fonction de Lorentz (PSF = 'l') : f(x) = a * c^2 / ( c^2 + (x - b)^2 )
			a :	Intensit�.
			b :	Position du pic
			c :	Largeur du pic = 1/2 largeur � 1/2 hauteur

		Param�tres pour la fonction Voigt (PSF = 'v', 'v2', 'v2k')
			k, beta_g, beta_l, I0g, I0l : voir [Langford1977]
			Pour PSF = 'v', ces 5 param�tres sont optimis�s
			Pour PSF = 'v2', on optimise 4 param�tres en posant I0g = I0l = I0lg
			Pour PSF = 'v2k', on mod�lise la s�paration des pics K-alpha1 et K-alpha2. Trait� de la m�me fa�on que 'v2' dans
			fit_approx, mais le traitement est diff�rent dans fit_calcul et dans iter.


		D�finition de la fonction de Lorentz interm�diaire (PSF = 'li') (implantation incompl�te) : 
		f(x) = a*sqrt( 4*(2^1.5-1) ) / (2*pi*c) * (1 + 4*( 2^(2/3) - 1 )*(x - b)^2/c^2)^(-1.5)
			a : Intensit�
			b : Position du pic
			c : FWHM

		Rajoute la liste des param�tres de r�gression (reg) dans la structure self.peak_list. Liste de la forme :
			[ [ PSF, a, b, c]
			  ...		 ]

			o� PSF est le type de fonction (e.g. 'g' pour gaussienne) et a, b et c les param�tres de la courbe

		"""

		if self.etat.peak_list == False:
			print 'Impossible de faire la r�gression, liste de pics manquante'
			return

		if not( PSF == 'g' or PSF =='l' or PSF == 'v' or PSF == 'v2' or PSF == 'v2k' ):
			print( u'Veuillez entrer une fonction de r�gression valide' )
			return

		I_max = 0

		if mode == 3:
			self.fit.R_vec = []
			self.fit.delta_vec = []

		for i in range( len( self.peak_list ) ):
			#Test pour voir si on mod�lise le pic i
			skip = True
			if mode == 1 and len( self.peak_list[i][1] ) == 0: #Aucune mod�lisation n'existe
				skip = False
			elif mode == 2 or mode == 4: #Mode 2 ou 4 et pic compris dans liste_pics
				for j in range( len(liste_pics) ):
					if self.peak_list[i][0] == liste_pics[j]:
						skip = False
			elif mode == 3:
				skip = False
			
			if skip == True:
				continue
			

			for j in range( len( self.data_smooth.theta ) ):
				if self.data_smooth.theta[j] >= self.peak_list[i][2]:
					break
			
			peak_index = j
			I = self.data_smooth.count[peak_index]
			if self.etat.back == True:
				I -= self.data_back.count[peak_index]

			I_max = max( I, I_max )
			th0 = self.data_smooth.theta[peak_index]
			
			#Trouve FWHM en trouvant le point o� I est la moiti� du sommet
			if self.etat.back == False:	
				while j < len( self.data_smooth.theta ) - 1 and self.data_smooth.count[j] > I / 2:
					j += 1
				
				HM_plus_index = j
				j = peak_index

				while j > 0 and self.data_smooth.count[j] > I / 2:
					j -= 1

				HM_moins_index = j
				
				if mode != 4:
					FWHM = self.data_smooth.theta[HM_plus_index] - self.data_smooth.theta[HM_moins_index]
					if FWHM > 2 and I < I_max/20:
						FWHM = 2

			#Si un bruit de fond est mod�lis�, on trouve FWHM avec la moiti� du sommet auquel le bruit de fond est soustrait
			else:
				while j < len( self.data_smooth.theta ) - 1 and (self.data_smooth.count[j] - self.data_back.count[j]) > I / 2:
					j += 1
				
				HM_plus_index = j
				j = peak_index

				while j > 0 and self.data_smooth.count[j] - self.data_back.count[j] > I / 2:
					j -= 1

				HM_moins_index = j
				
				if mode != 4:
					FWHM = self.data_smooth.theta[HM_plus_index] - self.data_smooth.theta[HM_moins_index]

			if mode == 4:
				FWHM = FWHM_appr
				I = I_appr


			if plot == 3:
				plt.plot( [th0 - FWHM/2, th0 + FWHM/2], [a/2, a/2] )
			
			if PSF == 'g':
				c = FWHM / (2*(2*np.log(2))**0.5)
			elif PSF == 'l':
				c = FWHM / 2
			elif PSF == 'li':
				c= FWHM

			elif PSF == 'v':
				k = 1/np.pi**0.5
				beta_g = ( FWHM**2*np.pi / (4.*(1.+k**2)) )**0.5
				beta_l = k * beta_g * np.pi**0.5
				I0g = (I / (beta_l * wofz( 1j*k )).real)**0.5
				I0l = I0g

			elif PSF == 'v2' or PSF == 'v2k':
				k = 1/np.pi**0.5
				beta_g = ( FWHM**2*np.pi / (4.*(1.+k**2)) )**0.5
				beta_l = k * beta_g * np.pi**0.5
				I0lg = (I / (beta_l * wofz( 1j*k )).real)**0.5


			if PSF == 'v':
				self.peak_list[i][1] = [PSF, I0g, I0l, th0, beta_g, beta_l]
			elif PSF == 'v2' or PSF == 'v2k':
				self.peak_list[i][1] = [PSF, I0lg, th0, beta_g, beta_l]
			else:	
				self.peak_list[i][1] = [PSF, I, th0, c]
			
			self.peak_list[i][3] = True
		
		self.etat.reg = True


	def split_Kalpha( self, no_seq, mat = mater.mat4340, emetteur = 'Cu', raie = 'a', plot = 0 ):
		"""
		� partir des param�tres de la PSF 'v2k', des propri�t�s du mat�riau et des rayons X incidents, retourne des fonctions lambda.
		Ces fonctions sont utilis�es seulement par fit_calcul, afin de tracer les fonctions et de calculer le r�sidu.
		La mod�lisation est bas�e sur un ratio d'intensit� I(K-alpha2)/I(K-alpha1) = 0.4457 (d'o� vient ce ratio???)
		La somme des surfaces des deux pics �gale la surface du pic global
		On consid�re que k1=k2=k, beta_g1=beta_g2=beta_g, beta_l1=beta_l2=beta_l3

		Args :
			no_seq : Num�ro s�quentiel du pic calcul�
			mat : Mat�riau de l'�chantillon
			emetteur, raie : Donn�es sur les rayons X incidents
			plot : 1 pour afficher les r�sultats

		Retourne :
			f : Fonction lambda de la somme des pics K-alpha1 et K-alpha2
			f1 : Fonction lambda du pic K-alpha1
			f2 : Fonction lambda du pic K-alpha2

		"""

		if self.etat.peak_list == False:
			print( u'Pic inexistant' )
			return

		for i in range( len( self.peak_list ) ):
			if self.peak_list[i][0] == no_seq:
				pic_index = no_seq

		PSF = self.peak_list[pic_index][1][0]
		phase_id = self.peak_list[pic_index][4]
		hkl = self.peak_list[pic_index][5]

		[mu_m, A, rho, lam] = xrdanal.read_melange( mat, emetteur, raie )
	
		if emetteur == 'Cu' and raie == 'a':
			lam_1 = 1.540562
			lam_2 = 1.544390

		
		if PSF == 'v2k':
			I0lg = self.peak_list[pic_index][1][1]
			t0 =  self.peak_list[pic_index][1][2]
			beta_g = self.peak_list[pic_index][1][3]
			beta_l = self.peak_list[pic_index][1][4]
			k = beta_l / (beta_g*np.pi**0.5)
			f = lambda x: ( beta_l*I0lg**2*wofz( np.pi**0.5*(x-t0)/beta_g + 1j*k )).real

		d = lam / ( 2*np.sin( t0/2. * 2.*np.pi/360. ) )

		#Pic k-alpha1 :
		t1 = 2.*360*np.arcsin( lam_1 / (2*d) )/(2.*np.pi)
		I0_1 = 0.8317 * I0lg
		f1 = lambda x: ( beta_l*I0_1**2*wofz( np.pi**0.5*(x-t1)/beta_g + 1j*k )).real

		#Pic k-alpha2 :
		t2 = 2.*360*np.arcsin( lam_2 / (2*d) )/(2.*np.pi)
		I0_2 = 0.5552 * I0lg
		f2 = lambda x: ( beta_l*I0_2**2*wofz( np.pi**0.5*(x-t2)/beta_g + 1j*k )).real
		
		if plot == 1:
			plt.plot( self.raw_data.theta, self.raw_data.count, label = r'Donn\'{e}es brutes', color = 'k' )
			plt.plot( self.raw_data.theta, f( self.raw_data.theta ), label = 'Pics confondus', color = 'b' )
			plt.plot( self.raw_data.theta, f1( self.raw_data.theta ), color = 'y', label = 'PSF individuelles' )
			plt.plot( self.raw_data.theta, f2( self.raw_data.theta ), color = 'y' )
			plt.plot( self.raw_data.theta, f1( self.raw_data.theta ) + f2( self.raw_data.theta ), color = 'r', label = r'Somme PSF s\'{e}par\'{e}es' )
			plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
			plt.ylabel( r'$I$ (comptes)' )
			plt.legend()
			plt.show()

		return f, f1, f2


	def fit_calcul( self, plot = 0, plotPSF = 0, emetteur = 'Cu', raie = 'a', mat = mater.mat4340 ):
		
		"""

		Calcule le spectre en fonction des param�tres de r�gression d�termin�s pr�c�demment
		Calcule le r�sidu
		Calcule R, R_p et R_wp
		
		Args :
			plot : Affiche la mod�lisation, les donn�es brutes et le r�sidu si plot = 1
			plotPSF : Si �gale 1, trace s�par�ment toutes les PSF
			emetteur, raie : Donn�es des rayons X incidents. Requis seulement si une ou plus des PSF est de type 'v2k'
			mat : Donn�es sur le mat�riau. Requis seulement si une ou plus des PSF est de type 'v2k'

		Stocke les donn�es dans la structure self.fit :
			self.fit.data_reg : 	Spectre calcul�
			self.fit.residu :	Spectre exp�rimental - Spectre calcul�
			self.fit.R
			self.fit.R_p
			self.fit.R_wp


		"""

		
		if self.etat.reg == False:
			print( u'Aucune r�gression' )
			return

		self.fit.data_reg = data()
		self.fit.data_reg.theta = copy.deepcopy( self.raw_data.theta )
		
		self.fit.residu = data()
		self.fit.residu.theta = copy.deepcopy ( self.raw_data.theta )
		self.fit.residu.count = 0*self.fit.residu.theta


		if plot == 1:
			f, axarr = plt.subplots( 2, sharex = True )
			axarr[0].plot( self.raw_data.theta, self.raw_data.count, label = r'Donn\'{e}es brutes' )


		#Si le bruit de fond est calcul�, il est ajout� � la mod�lisation
		if self.etat.back == True:
			self.fit.data_reg.count = copy.deepcopy( self.data_back.count )
			if plotPSF == 1 and plot == 1:
				axarr[0].plot( self.data_back.theta, self.data_back.count, color = 'c', label = 'Bruit de fond' )
				axarr[0].plot( [], [], color = 'y', label = 'PSF individuelles' )
		else:
			self.fit.data_reg.count = 0*self.fit.data_reg.theta


		for i in range( len( self.peak_list ) ):
			if len( self.peak_list[i][1] ) == 0:
				continue

			PSF = self.peak_list[i][1][0]
			I = self.peak_list[i][1][1]
			th0 = self.peak_list[i][1][2]
			c = self.peak_list[i][1][3]
			
			if PSF == 'g':
				f = lambda x: I*np.exp( -(x - th0)**2/(2*c**2) )

			elif PSF == 'l':
				f = lambda x: I*c**2 / (c**2 + (x - th0)**2)

			elif PSF == 'li':
				f = lambda x: I*np.sqrt(4.*(2.**1.5-1.)) / (2.*np.pi*c) * (1. + 4.*(2.**(2./3.)-1.) / c**2. * (x - th0)**2.)**(-1.5)

			elif PSF == 'v':
				I0g = self.peak_list[i][1][1]
				I0l = self.peak_list[i][1][2]
				t0 =  self.peak_list[i][1][3]
				beta_g = self.peak_list[i][1][4]
				beta_l = self.peak_list[i][1][5]
				k = beta_l / (beta_g*np.pi**0.5)
				f = lambda x: ( beta_l*I0l*I0g*wofz( np.pi**0.5*(x-t0)/beta_g + 1j*k )).real

			elif PSF == 'v2':
				I0lg = self.peak_list[i][1][1]
				t0 =  self.peak_list[i][1][2]
				beta_g = self.peak_list[i][1][3]
				beta_l = self.peak_list[i][1][4]
				k = beta_l / (beta_g*np.pi**0.5)
				f = lambda x: ( beta_l*I0lg**2*wofz( np.pi**0.5*(x-t0)/beta_g + 1j*k )).real

			elif PSF == 'v2k':
				
				[f, f1, f2] = self.split_Kalpha( self.peak_list[i][0], mat = mat, emetteur = emetteur, raie = raie )

			if plotPSF == 1 and plot == 1:
				if PSF == 'v2k':
					axarr[0].plot( self.fit.data_reg.theta, f1( self.fit.data_reg.theta ), color = 'y' )
					axarr[0].plot( self.fit.data_reg.theta, f2( self.fit.data_reg.theta ), color = 'y' )
				else:	
					axarr[0].plot( self.fit.data_reg.theta, f( self.fit.data_reg.theta ), color = 'y' )

			if PSF == 'v2k':
				self.fit.data_reg.count += f1( self.fit.data_reg.theta )
				self.fit.data_reg.count += f2( self.fit.data_reg.theta )
			else:
				self.fit.data_reg.count += f( self.fit.data_reg.theta )

		s_res_2 = 0
		s_iobs_2 = 0
		s_w_iobs_2 = 0
		s_abs_res = 0
		s_iobs = 0
		s_w_res_2 = 0

		for i in range( len( self.fit.data_reg.theta ) ):
			i_obs = self.raw_data.count[i]
			i_calc = self.fit.data_reg.count[i]
			w = 1./i_obs
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


		self.fit.R = 100*( s_res_2/s_iobs_2 )**0.5	#[Howard1989]
		self.fit.R_p = 100*s_abs_res/s_iobs		#[Parrish2004]
		self.fit.R_wp = 100*( s_w_res_2/s_w_iobs_2 )**0.5#[Parrish2004]
		self.est_noise()

		self.etat.calcul = True

		if plot == 1:
			print( 'R = \t' + str( np.round( self.fit.R, 3 ) ) + ' %' )
			print( 'R_p = \t' + str( np.round( self.fit.R_p, 3 ) ) + ' %' )
			print( 'R_wp = \t' + str( np.round( self.fit.R_wp, 3) ) + ' %' )

			axarr[0].plot( self.fit.data_reg.theta, self.fit.data_reg.count, label = r'R\'{e}gression' )
			axarr[0].set_ylabel( r'$I$ (comptes)' )
			axarr[0].legend()
			axarr[1].plot( self.fit.residu.theta, self.fit.residu.count, label = r'R\'{e}sidu' )
			axarr[1].set_ylabel( r'$I$ (comptes)' )
			axarr[1].set_xlabel( r'$2 \theta$ ($^\circ$)' )
			axarr[1].legend()

			plt.show()
				

	def trace_fit( self ):			
		"""
		Trace les donn�es brutes et la r�gression
		"""
		trace( self.raw_data, plot = 2, lbl = r'Donn\'{e}es brutes' )
		trace( self.data_reg, plot = 2, lbl = r'R\'{e}gression' )
		
		plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
		plt.ylabel( r'$I$ (comptes)' )
		plt.legend()
		plt.show()

	def background_find_peaks( self, plot = 0 ):
		"""
		Cr�e une liste des points m�dians entre les pics, afin de faire une approximation du background

		Args :
			plot : Mettre � 1 pour afficher les r�sultats

		Rajoute la liste de points dans la structure self.back_pts_list
		"""
		if self.etat.peak_list == False:
			print( u'�tablir la liste des pics en premier' )
			return

		peak_list_sorted = []
		self.back_pts_list = []
		f_count = interp1d( self.data_smooth.theta, self.data_smooth.count )

		for i in range( len( self.peak_list ) ):
			peak_list_sorted.append( self.peak_list[i][2] )

		peak_list_sorted.sort()	

		self.back_pts_list.append( [self.data_smooth.theta[0], float( f_count(self.data_smooth.theta[0]) ) ] )
		for i in range( len( peak_list_sorted ) - 1):
			theta_mid = (peak_list_sorted[i] + peak_list_sorted[i+1])/2 
			self.back_pts_list.append( [theta_mid, float( f_count(theta_mid) )] )

		self.back_pts_list.append( [self.data_smooth.theta[-1], float( f_count(self.data_smooth.theta[-1]) ) ] )
		self.etat.back_list = True

		if plot == 1 or plot ==2:
			plt.plot( self.raw_data.theta, self.raw_data.count, label = r'Donn\'{e}es brutes' )
			plt.plot( self.data_smooth.theta, self.data_smooth.count, label = r'Donn\'{e}es liss\'{e}es, n = 3' )
			for i in range( len( self.back_pts_list ) ):
				plt.plot( self.back_pts_list[i][0], self.back_pts_list[i][1], 'ro' )
			
			if plot == 1:
				plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
				plt.ylabel( r'$I$ (comptes)' )
				plt.legend()
				plt.show()

	def background_approx( self, plot = 0, dist = 2, method = '1d' ):
		"""

		Mod�lise le background

		Args :
			dist :	Argument pour la m�thode de moyenne (voir ci-dessous)
			method : M�thode pour calculer. Diff�rents choix :
				
				'1d' : � partir d'une liste de points obtenue pr�c�demment et contenue dans la structure
				self.back_pts_list, obtenue par exemple avec la fonction background_find_peaks, mod�lise
				le background comme une interpolation 1d de ces points.
				

		Le r�sultat de la mod�lisation est stock� dans la structure self.data_back.		

		"""
		
		if self.etat.peak_list == False:
			print( u'�tablir la liste des pics en premier' )
			return
		
		self.data_back = data()


		self.data_back.theta = copy.deepcopy( self.raw_data.theta )
		mult = np.zeros( len( self.data_back.theta ) )

		if method == '1d':
			if self.etat.back_list == False:
				print( u'Ex�cuter background_find_peaks en premier' )
				return

			liste_theta_back = []
			liste_count_back = []
			for i in range( len( self.back_pts_list ) ):
				liste_theta_back.append( self.back_pts_list[i][0] )
				liste_count_back.append( self.back_pts_list[i][1] )
			f1d = interp1d( liste_theta_back, liste_count_back, bounds_error = False, fill_value = 'extrapolate' )
			self.data_back.count = f1d( self.data_back.theta )


		self.etat.back = True

		if plot == 3:
			plt.plot( self.data_back.theta, self.data_back.count )
			plt.show()

	def iter( self, nit = 10, logres = 0, alpha = 1, alpha_back = 1, gamma = 0, plot = 0, track = 1, modback = False, ret_J = False, emetteur = 'Cu', raie = 'a', mat = mater.mat4340 ):
		"""
		Raffine la solution calcul�e � l'aide de la m�thode des moindres carr�s non-lin�aires
		L'algorithme de Gauss-Newton est utilis� de fa�on it�rative pour calculer la solution approch�e
		Le syst�me d'�quations lin�aires est r�solu � chaque it�ration par d�composition de la matrice
		des coefficients en facteurs de Cholesky et r�solution de deux syst�mes triangulaires.
	
		Args :
			nit :		Nombre d'it�rations � effectuer
			logres :	Ordre de grandeur du r�sidu voulu avant d'arr�ter le calcul.
					Pour un calcul qui s'arr�te seulement sur un crit�re de r�sidu pour un 
					nombre illimit� d'it�rations, choisir nit=0.
					logres=0 d�sactive ce crit�re
			alpha :		Facteur multiplicatif du r�sidu (1 par d�faut), "shift-cutting". Choisir une valeur inf�rieure peut
					am�liorer la convergence de l'algorithme. [Wikipedia]
			alpha_back :	Pour acc�l�rer la convergence de l'arri�re-plan, ce facteur sera multipli� au facteur alpha pour le
					la changement dans le cas de l'arri�re-plan.
			gamma :		Param�tre de Marquadt. Additionne une matrice identit� multipli�e par ce facteur � la matrice des
					coefficients. Modifie la direction de descente et peut am�liorer la convergence si le "shift cutting"
					n'est pas efficace.
			plot :		plot = 1 affiche le r�sultat de la mod�lisation
			track :		Track = 1 affiche le progr�s du calcul � chaque it�ration et montre la courbe de convergence � la fin
					du calcul.
			modback :	Mettre � <True> pour raffiner l'intensit� du bruit de fond en m�me temps que la corr�lation
			ret_J :		Retourne le Jacobien (mode d�bogage)
			emetteur, raie :Donn�es sur les rayons X incidents, requis dans si une ou plus des PSF est 'v2k'
			mat :		Mat�riau, requis si une ou plus des PSF est 'v2k'.

		L'algorithme met � jour la mod�lisation dans la structure self.peak_list.reg et fait appel � la routine fit_calcul pour
		trouver la solution � chaque point ainsi que le r�sidu utilis� dans l'it�ration suivante.
		"""

		[mu_m, A, rho, lam] = xrdanal.read_melange( mat, emetteur, raie )
		if emetteur == 'Cu' and raie == 'a':
			lam_1 = 1.540562
			lam_2 = 1.544390

		if self.etat.reg == False:
			print( u'Aucune r�gression')
			return
		
		self.fit_calcul()

		n = len( self.raw_data.theta )
		
		#Construit le dictionnaire reliant la colonne du Jacobien avec le bon param�tre
		list_args = {}
		J_index = 0
		for i in range( len( self.peak_list ) ):
			if self.peak_list[i][3] == True:
				if self.peak_list[i][1][0] == 'v':
					list_args[J_index] = 'p' + str(self.peak_list[i][0]) + '1'
					list_args[J_index + 1] = 'p' + str(self.peak_list[i][0]) + '2'
					list_args[J_index + 2] = 'p' + str(self.peak_list[i][0]) + '3'
					list_args[J_index + 3] = 'p' + str(self.peak_list[i][0]) + '4'
					list_args[J_index + 4] = 'p' + str(self.peak_list[i][0]) + '5'
					J_index += 5
				elif self.peak_list[i][1][0] == 'v2' or self.peak_list[i][1][0] == 'v2k':
					list_args[J_index] = 'p' + str(self.peak_list[i][0]) + '1'
					list_args[J_index + 1] = 'p' + str(self.peak_list[i][0]) + '2'
					list_args[J_index + 2] = 'p' + str(self.peak_list[i][0]) + '3'
					list_args[J_index + 3] = 'p' + str(self.peak_list[i][0]) + '4'
					J_index += 4
				else:
					list_args[J_index] = 'p' + str(self.peak_list[i][0]) + '1'
					list_args[J_index + 1] = 'p' + str(self.peak_list[i][0]) + '2'
					list_args[J_index + 2] = 'p' + str(self.peak_list[i][0]) + '3'
					J_index += 3

		if modback == True and self.etat.back_list == True:
			for i in range( len( self.back_pts_list ) ):
				list_args[J_index] = 'b' + str(i)
				J_index += 1
		
		m = len( list_args )
		J = np.zeros( (n, m) )

		k = 0	#Compteur d'it�rations
		erreur = 0
		
		if track == 1:
			print('It.\tR\t\tdelta_beta/beta')
			print('0\t' + str( self.fit.R ) )
		
		#Boucle principale de calcul
		try:
			while (k < nit or nit == 0):
				#Constitution de la matrice jacobienne � l'aide des d�riv�es partielles des PSF
				for i in range( n ):
					for j in range( m ):
						cle = list_args[j]

						if cle[0] == 'p':
							#L'argument est un pic
							th = self.raw_data.theta[i]
							pic = int(cle[1:-1])
							arg = int(cle[-1])
							for l in range( len( self.peak_list ) ):
								if self.peak_list[l][0] == pic:
									pic_index = l
							
							PSF = self.peak_list[pic_index][1][0]


							if PSF == 'g' or PSF == 'l':
								I = self.peak_list[pic_index][1][1]
								th0 = self.peak_list[pic_index][1][2]
								c = self.peak_list[pic_index][1][3]

							if PSF == 'g':
								if arg == 1:
									J[i, j] = np.exp( -(th-th0)**2 / (2*c**2) )
								elif arg == 2:	
									J[i, j] = I*(th - th0)/(c**2) * J[i, j-1]
								elif arg == 3:
									J[i, j] = I*(th - th0)**2/(c**3) * J[i, j-2]

							elif PSF == 'l':
								if arg == 1:
									J[i, j] = c**2 / (c**2 + (th - th0)**2) 
								elif arg == 2:	
									J[i, j] = 2*I*c**2*(th - th0) / (c**2 + (th - th0)**2 )**2 
								elif arg == 3:	
									J[i, j] = 2*I*c*(th - th0)**2 / (c**2 + (th - th0)**2 )**2 

							elif PSF == 'v':
								I0g = self.peak_list[pic_index][1][1]
								I0l = self.peak_list[pic_index][1][2]
								th0 = self.peak_list[pic_index][1][3]
								beta_g = self.peak_list[pic_index][1][4]
								beta_l = self.peak_list[pic_index][1][5]
								k_voigt = beta_l / (beta_g*np.pi**0.5)
								z = np.pi**0.5*(th - th0)/beta_g + 1j*k_voigt
								I = beta_l*I0g*I0l*wofz(z).real
								dwdz = -2*z*wofz(z) + 2j/np.pi**0.5
								if arg == 1:
									J[i, j] = I/I0g
								elif arg == 2:	
									J[i, j] = I/I0l
								elif arg == 3:	
									J[i, j] = -beta_l*I0l*I0g*np.pi**0.5/beta_g*( dwdz ).real
								elif arg == 4:	
									J[i, j] = beta_l*I0l*I0g*( -dwdz*z/beta_g**2 ).real
								elif arg == 5:
									J[i, j] = I/beta_l + beta_l*I0l*I0g*( dwdz*1j/(beta_g*np.pi**0.5) ).real

							elif PSF == 'v2':
								I0lg = self.peak_list[pic_index][1][1]
								th0 = self.peak_list[pic_index][1][2]
								beta_g = self.peak_list[pic_index][1][3]
								beta_l = self.peak_list[pic_index][1][4]
								k_voigt = beta_l / (beta_g*np.pi**0.5)
								z = np.pi**0.5*(th - th0)/beta_g + 1j*k_voigt
								I = beta_l*I0lg**2*wofz(z).real
								dwdz = -2*z*wofz(z) + 2j/np.pi**0.5
								if arg == 1:
									J[i, j] = 2*I/I0lg
								elif arg == 2:	
									J[i, j] = -beta_l*I0lg**2*np.pi**0.5/beta_g*( dwdz ).real
								elif arg == 3:	
									J[i, j] = beta_l*I0lg**2*( -dwdz*z/beta_g ).real
								elif arg == 4:
									J[i, j] = I/beta_l + beta_l*I0lg**2*( dwdz*1j/(beta_g*np.pi**0.5) ).real

							elif PSF == 'v2k':
								I0lg = self.peak_list[pic_index][1][1]
								th0 = self.peak_list[pic_index][1][2]
								beta_g = self.peak_list[pic_index][1][3]
								beta_l = self.peak_list[pic_index][1][4]
								k_voigt = beta_l / (beta_g*np.pi**0.5)
								
								d = lam / ( 2*np.sin( th0/2. * 2.*np.pi/360. ) )
								
								delta_th = 2.*360./(2*np.pi) * ( np.arcsin( lam_1 / (2*d) ) - np.arcsin( lam_2 / (2*d)))

								z1 = np.pi**0.5*(th - th0 - 0.2*delta_th)/beta_g + 1j*k_voigt
								z2 = np.pi**0.5*(th - th0 + 0.8*delta_th)/beta_g + 1j*k_voigt

								I1 = 0.6917*beta_l*I0lg**2*wofz(z1).real
								I2 = 0.3083*beta_l*I0lg**2*wofz(z2).real
								I = I1 + I2

								dwdz1 = -2*z1*wofz(z1) + 2j/np.pi**0.5
								dwdz2 = -2*z2*wofz(z2) + 2j/np.pi**0.5

								if arg == 1:
									J[i, j] = 2*I/I0lg
								elif arg == 2:	
									fact1 = -beta_l*I0lg**2*np.pi**0.5/beta_g
									fact2 = lam * np.cos(th0/2. * 2*np.pi/360.) / (2*(np.sin(th0/2. * 2*np.pi/360.)**2))
									J1 = (1 + lam_2/10/(1-lam_2**2/4/d**2)**0.5 - lam_1/10/(1-lam_1**2/4/d**2)**0.5)
									J2 = (1 - 2*lam_2/5/(1-lam_2**2/4/d**2)**0.5 + 2*lam_1/5/(1-lam_1**2/4/d**2)**0.5)
									J[i, j] = fact1*fact2*( 0.6917 * J1 * dwdz1.real + 0.3083 * J2 * dwdz2.real )
								elif arg == 3:	
									J1 = 0.6917*beta_l*I0lg**2*( -dwdz1*z1/beta_g ).real
									J2 = 0.3083*beta_l*I0lg**2*( -dwdz2*z2/beta_g ).real
									J[i, j] = J1 + J2

								elif arg == 4:
									J1 = 0.6917*beta_l*I0lg**2*( dwdz1*1j/(beta_g*np.pi**0.5) ).real
									J2 = 0.3083*beta_l*I0lg**2*( dwdz2*1j/(beta_g*np.pi**0.5) ).real
									J[i, j] = I/beta_l + J1 + J2

						elif cle[0] == 'b':
							#L'argument est un point de bruit de fond
							pt_no = int( cle[1:] )
							th_j = self.back_pts_list[pt_no][0]
							th = self.raw_data.theta[i]
							
							if pt_no == 0:
								th_j_plus = self.back_pts_list[pt_no+1][0]
								if th < th_j_plus:
									J[i, j] = 1 - (th - th_j)/(th_j_plus - th_j)
								else:
									J[i, j] = 0
							elif pt_no == len( self.back_pts_list ) - 1:
								th_j_moins = self.back_pts_list[pt_no-1][0]
								if th > th_j_moins:
									J[i, j] = (th - th_j_moins) / (th_j - th_j_moins)
								else:
									J[i, j] = 0
							else:
								th_j_plus = self.back_pts_list[pt_no+1][0]
								th_j_moins = self.back_pts_list[pt_no-1][0]
								if th > th_j and th < th_j_plus:
									J[i, j] = 1 - (th - th_j)/(th_j_plus - th_j)
								elif th < th_j and th > th_j_moins:
									J[i, j] = (th - th_j_moins)/(th_j - th_j_moins)
								else:
									J[i, j] = 0

			
				#Calcul de la matrice des coefficients (JtJ) et du vecteur des solutions (JtR)
				Jt = np.transpose( J )
				JtJ = np.matmul( Jt, J ) + gamma * np.eye( m )
				JtR = np.matmul( Jt, self.fit.residu.count )
				try:
					#Factorisation de Cholesky et r�solution du syst�me lin�aire
					L = np.linalg.cholesky( JtJ )
					y = solve_triangular( L, JtR, lower=True )
					delta_beta = solve_triangular( np.transpose(L), y )
				except KeyboardInterrupt:
					raise KeyboardInterrupt
				except:
					print( 'Erreur, divergence' )
					erreur = 1
					break
					

				#Reconstitue le vecteur delta
				norm_beta = 0
				norm_delta_beta = 0
				for j in range( m ):
					cle = list_args[j]
					if cle[0] == 'p':
						pic = int(cle[1:-1])
						arg = int(cle[-1])
						for l in range( len( self.peak_list ) ):
							if self.peak_list[l][0] == pic:
								pic_index = l

						delta_arg = delta_beta[j]
						self.peak_list[pic_index][1][arg] += alpha * delta_arg
						norm_beta += self.peak_list[pic_index][1][arg]**2
					
					elif cle[0] == 'b':
						pt_no = int( cle[1:] )
						delta_arg = delta_beta[j]
						self.back_pts_list[pt_no][1] += alpha * alpha_back * delta_arg
						norm_beta += self.back_pts_list[pt_no][1]**2

					norm_delta_beta += delta_arg**2
						



				crit2 = np.sqrt(norm_delta_beta/norm_beta)

				#Calcul de la nouvelle approximation du bruit de fond, de la solution et du r�sidu
				self.background_approx()
				self.fit_calcul()
				
				#Mise � jour et affichage des donn�es de l'�volution du calcul it�ratif
				k += 1
				self.fit.R_vec = np.append( self.fit.R_vec, self.fit.R )
				self.fit.delta_vec = np.append( self.fit.delta_vec, crit2 )
				if track == 1:
					print(str(k) + '\t' + str(self.fit.R) + '\t' + str(crit2) )

				#Test de convergence du param�tre logres
				if logres != 0 and crit2 < 10**logres:
					break

		except KeyboardInterrupt:
			pass

		if track == 1:
			f, axarr = plt.subplots( 2, sharex = True )
			axarr[0].plot( np.arange( len(self.fit.R_vec) ) + 1, self.fit.R_vec )
			axarr[0].set_ylabel( r'$R$' )
			axarr[1].plot( np.arange( len(self.fit.delta_vec) ) +1, np.log10(self.fit.delta_vec) )
			axarr[1].set_xlabel( r'It\'{e}ration' )
			axarr[1].set_ylabel( r'log( r\'{e}sidu )' )
			plt.show()

		if plot == 1:	
			self.fit_calcul( plot=1 )

		if erreur == 1:
			return J

		if ret_J == True:
			return J
		

	def est_noise( self, plot = 0 ):
		"""
		Estime l'amplitude du bruit.

		M�thode : 
			1. En utilisant la r�gression, cible les zones � l'ext�rieur des pics, par le crit�re voulant que le signal
			   d'arri�re-plan soit sup�rieur � la somme de toutes les PSF.
			2. Calcule l'�cart-type du r�sidu calcul� dans ces zones.   

	
		"""
		
		if self.etat.reg == False or self.etat.back == False:
			print( u'Faire une r�gression en premier lieu et calculer l\'arri�re-plan' )
			return

		is_back = data()
		is_back.theta = copy.deepcopy( self.raw_data.theta )

		is_back.count = self.data_back.count > ( self.fit.data_reg.count - self.data_back.count )

		if plot == 1:
			plt.plot( is_back.theta, is_back.count*self.fit.residu.count )
			plt.show()
			
		noise_vec = np.array([])	
		for i in range( len( self.raw_data.count) ):
			if is_back.count[i] == True:
				noise_vec = np.append( noise_vec, self.fit.residu.count[i] )
		
		if plot == 1:
			plt.plot( noise_vec )
			plt.show()
		
		self.fit.noise = np.std( noise_vec )
		
		
def lisser_moy( data_in, n = 1, plot = 0, lbl = '' ):
	"""
	Effectue un lissage des donn�es
	Avec la m�thode de la moyenne mobile avec n donn�es adjacentes

	Args :
		data_in :	Donn�es initiales. Doit �tre de classe data
		ordre :		Ordre de pr�cision du sch�ma num�rique
		plot :		0 : Aucun graphe
				1 : Trace imm�diatement
				2 : Conserve en m�moire
		lbl :		�tiquette des donn�es si plot = 2

	Retourne :
		data_out :	D�riv�e, sous forme de classe data
	"""
	
	data_out = data()
	for i in range( len( data_in.theta ) - 2*n ):
		data_out.theta = np.append( data_out.theta, data_in.theta[i + n] )
		data_out.count = np.append( data_out.count, data_in.count[i + n] )

		j = n
		while j >= 1:
			data_out.count[i] += data_in.count[i + n - j] + data_in.count[i + n + j]
			j -= 1
		
		data_out.count[i] = data_out.count[i] / (2*n + 1)

	if plot == 1 or plot ==2:
		plt.plot(data_out.theta, data_out.count, label = lbl )
		
		if plot == 1:
			plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
			plt.ylabel( r'$I$ (comptes)' )
			plt.show()

	return data_out		

def diff_data( data_in, n = 1, plot = 0, lbl = '' ):
	"""
	Calcule la d�riv�e d'une s�rie de donn�es ( dI / d(theta) )
	D�riv�e num�rique par sch�ma centr� de diff�rences finies d'ordre 2*n

	Args :
		data_in :	Donn�es initiales. Doit �tre de classe data
		ordre :		Ordre de pr�cision du sch�ma num�rique
		plot :		0 : Aucun graphe
				1 : Trace imm�diatement
				2 : Conserve en m�moire
		lbl :		�tiquette des donn�es si plot = 2

	Retourne :
		data_out :	D�riv�e, sous forme de classe data
	"""

	data_out = data()
	if n == 1:
		coeffs = [-0.5, 0, 0.5]
	elif n == 2:
		coeffs = [1./12, -2./3, 0, 2./3, -1./12]
	elif n == 3:
		coeffs = [-1./60, 3./20, -.3/4, 0, 3/4, -3./20, 1./60]
	elif n == 4:
		coeffs = [1./280, -4./105, 1./5, -4./5, 0, 4./5, -1./5, 4./105, -1./280]
	else:
		print( u"Argument <ordre> : entrer un chiffre de 1 � 4" )
		return

	h = data_in.theta[1] - data_in.theta[0]

	for i in range( len( data_in.theta ) - 2*n ):
		data_out.theta = np.append( data_out.theta, data_in.theta[i + n] )
		data_out.count = np.append( data_out.count, 0. )

		j = n
		while j >= 1:
			data_out.count[i] += data_in.count[i + n - j]*coeffs[n - j]*h
			data_out.count[i] += data_in.count[i + n + j]*coeffs[n + j]*h
			j -= 1

	if plot == 1 or plot ==2:
		plt.plot(data_out.theta, data_out.count, label = lbl )
		
		if plot == 1:
			plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
			plt.ylabel( r'$\frac{dI}{d \theta}$ (comptes)' )
			plt.show()

	return data_out


def trace( data_in, abscisse = 0, ordonnee = 0, plot = 1, lbl = '' ):
	"""
	Trace un spectre � partir de donn�es obtenues dans diff�rentes fonctions de ce programme

	Args:
		data_in : Donn�es � tracer (classe data).
		abscisse : 
			0 : Donn�es brutes
		
		ordonnee : 
			0 : Comptes
			1 : Comptes**0.5
		
		plot :
			1 : Tracer imm�diatement
			2 : Conserver en m�moire
		
		label :	�tiquette de la s�rie de donn�es si graphe conserv� en m�moire
	"""

	if len( data_in.theta ) == 0:
		print( u'Aucune donn�e' )
		return

	if plot == 1:
		plt.figure()

	if abscisse == 0 and ordonnee == 0:
		plt.plot( data_in.theta, data_in.count, label=lbl )
		plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
		plt.ylabel( r'$I$ (comptes)' )

	elif abscisse == 0 and ordonnee == 1:
		plt.plot( data_in.theta, data_in.count**0.5, label=lbl )
		plt.xlabel( r'$2 \theta$ ($^\circ$) ' )
		plt.ylabel( r'$I^{0.5}$ (comptes)$^{0.5}$' )

	if plot == 2 and lbl != '':
		plt.legend()

	if plot == 1:
		plt.show()

