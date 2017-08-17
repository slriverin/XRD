# -*- coding: latin-1 -*-

mat4340 = [	[ '4340-std', 'm'	],	#4340 sur lequel aucune analyse chimique n'est faite
		[ 'Ni'	, '0.018(1)'	],	#On présume qu'il suit le standard et la valeur attendue est
		[ 'Cr'	, '0.0080(5)'	],	#située au centre de l'intervalle. Pour calculer l'écart type,
		[ 'Mn'	, '0.0078(7)'	],	#on considère une distribution uniforme
		[ 'Mo'	, '0.0025(3)'	],
		[ 'Si'	, '0.0025(5)'	],
		[ 'C'	, '0.0041(1)'	],
		[ 'Fe'	, 'B'		]]

SS310 = [	[ 'SS310 - SRM488', 'm'	],	#Selon analyse chimique fournie dans SP260-86 sur le standard SRM488
		[ 'Cr'	, '0.2499(1)'	],	#Les écarts-types ne sont pas donnés dans le document alors on considère de
		[ 'Ni'	, '0.2041(1)'	],	#façon conservatrice une valeur de 1 sur le dernier chiffre significatif
		[ 'C'	, '0.0005(1)'	],
		[ 'Mn'	, '0.0020(1)'	],
		[ 'Si'	, '0.0075(1)'	],
		[ 'Fe'	, 'B'		]]

SS430 = [	[ 'SS430 - SRM488', 'm'	],
		[ 'Cr'	, '0.1682(1)'	],
		[ 'Ni'	, '0.0008(1)'	],
		[ 'C'	, '0.0002(1)'	],
		[ 'Mn'	, '0.0015(1)'	],
		[ 'Si'	, '0.0083(1)'	],
		[ 'Fe'	, 'B'		]]
