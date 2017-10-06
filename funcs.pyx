cimport scipy.special.cython_special as csc
from scipy.integrate import quad
import numpy as np
cimport numpy as np
from cpython cimport array
import array

cdef extern from "math.h":
	double cos "cos" (double)
	double acos "acos" (double)

import numpy as np

def def_f_pic(double I0lg, double theta_obs, double beta_g, double beta_l):
	cdef double k
	k = beta_l/np.pi**0.5/beta_g
	def f_pic( double theta ):
		return beta_l*I0lg**2*(csc.wofz(np.pi**0.5*(theta-theta_obs)/beta_g + 1j*k)).real

	return f_pic

cdef double f_h( double phi, double th_obs, double L ):
	try:
		return L*( ( cos(phi*2*np.pi/360)/cos(th_obs*2*np.pi/360) )**2 - 1)**0.5
	except ValueError:
		return 0

cdef double f_W( double phi, double th_obs, double phi_min, double phi_infl, double H, double S, double L ):
	if phi < phi_min:
		return 0
	elif phi > th_obs:
		return 0
	elif phi < phi_infl:
		return H + S - f_h(phi, th_obs, L)
	else:
		return 2*S

cdef double f_D( double phi, double th_obs, double phi_min, double phi_infl, double H, double S, double L ):
	cdef double W
	if phi < phi_min:
		return 0
	elif phi > th_obs:
		return 0
	elif phi < phi_infl:
		W = H + S - f_h(phi, th_obs, L)
	else:
		W = 2*S
	try:
		return L*W/(2*H*f_h(phi, th_obs, L)*cos(phi*2*np.pi/360))
	except ZeroDivisionError:
		return 0

cdef double intD( double th_obs, double phi_min, double phi_infl, double H, double S, double L):
	cdef double th0 = th_obs*2*np.pi/360
	cdef double a = 0.5*(th0 + np.pi/2)
	
	#Intégrale contenant fonction W pour th0>thinfl, évaluée avec la singularité par la méthode des résidus
	def f_res(t):
		return (1j*a*np.exp(1j*t)/(np.cos(a*np.exp(1j*t))*( (np.cos(a*np.exp(1j*t)))**2/(np.cos(th0))**2 - 1)**0.5 )).real
	I_res = -np.min([H, S])/H*quad(f_res, 0, np.pi/2)[0]

	#On doit soustraire à l'intégrale ci-dessus pour conserver seulement la plage entre l'inflexion et le max
	def f_soustr(t):
		return 1./(cos(t)*( (cos(t))**2/(cos(th0))**2 - 1)**0.5)
	I_soustr = np.min([H, S])/H*quad( f_soustr, 0, phi_infl*2*np.pi/360 )[0]

	#On rajoute ensuite le reste de la place, soit phi_min<phi<phi_infl
	def f_min_infl(t):
		return f_W(t*360./2/np.pi, th_obs, phi_min, phi_infl, H, S, L )/(2*H*cos(t)*( (cos(t))**2/(cos(th0))**2 - 1)**0.5)
	
	I_min_infl = quad(f_min_infl, phi_min*2*np.pi/360, phi_infl*2*np.pi/360)[0]
	
	return (I_res - I_soustr + I_min_infl)*360/2/np.pi

def corr_asym( theta_range, double theta_obs, double H, double S, double L, double cutoff, f_pic ):
	
	cdef double phi_min = acos( cos( theta_obs*2*np.pi/360)*(((H+S)/L)**2+1)**0.5 )*360/2/np.pi
	cdef double phi_infl = acos( cos( theta_obs*2*np.pi/360)*(((H-S)/L)**2+1)**0.5 )*360/2/np.pi
	cdef int n = len( theta_range )
	count_temp = np.zeros( n )

	if theta_obs < 90:
		surf_D = intD(theta_obs, phi_min, phi_infl, H, S, L)
		for j in range( n ):
			phi = theta_range[j]
			if f_pic(phi) < cutoff:
				continue
			fd = lambda t : f_D(t, theta_obs, phi_min, phi_infl, H, S, L)
			f_asym = lambda t: fd(t)*f_pic(theta_obs+phi-t)/surf_D
			count_temp[j] = quad(f_asym, phi_min, theta_obs)[0]
			
	else:
		surf_D = intD(180. - theta_obs, 180. - phi_min, 180. - phi_infl, H, S, L)
		for j in range( n ):
			phi = theta_range[j]
			if f_pic(phi) < cutoff:
				continue
			fd = lambda t : f_D(t, 180. - theta_obs, 180. - phi_min, 180. - phi_infl, H, S, L)
			f_asym = lambda t: fd(180.-t)*f_pic(theta_obs+phi-t)/surf_D
			count_temp[j] = quad(f_asym, theta_obs, phi_min)[0]

	return count_temp
