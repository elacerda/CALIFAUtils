# Morfologia, incluyendo tipo medio y +- error
# ES.Enrique . DF . 20120808
# ES.Enrique . Chiclana . 20140417 . Corrected to distinguish E0 and S0.

import numpy as np

# indir = '/Users/cid/PycharmProjects/DeVolta/'

def get_morfologia(Kname) : 
	Korder = int(Kname[1:])
	# lee el numero de la galaxia, tipo y subtipo morfologico	
	id, name, morf0,morf1, morf_m0,morf_m1, morf_p0,morf_p1, bar, bar_m, bar_p = np.loadtxt('/Users/lacerda/CALIFA/morph_eye_class.csv', delimiter=',',unpack=True,usecols=(0,2,5,6,7,8,9,10,12,13,14),skiprows=23,dtype={'names': ('a','b','c','d','e','f','g','h','i','j','k'),'formats': ('I3','S15','S3','S3','S3','S3','S3','S3','S3','S3','S3')})
	morf = [morf0[i].strip()+morf1[i].strip() for i in range(len(morf0))]
	morf_m = [morf_m0[i].strip()+morf_m1[i].strip() for i in range(len(morf0))]
	morf_p = [morf_p0[i].strip()+morf_p1[i].strip() for i in range(len(morf0))]

	# convierte tipo y subtipo morfologico a valor numerico T (-7:E0 -1:E7 0:S0 5:Sm) en array 'tipo'
	# este algoritmo es una verdadera chapuza, pero funciona.
	type = [['E0','E1','E2','E3','E4','E5','E6','E7','S0','S0a','Sa','Sab','Sb','Sbc','Sc','Scd','Sd','Sdm','Sm','Ir'], \
			[  0,   1,   2,   3,   4,   5,   6,   7,   8,  8.5,   9,  9.5,  10,  10.5, 11, 11.5,  12, 12.5,  13,  14]]
	
	tipos = morf[Korder-1]        # tipo medio ascii
	tipo = type[1][type[0].index(morf[Korder-1])]        # tipo medio
	tipo_m = type[1][type[0].index(morf_m[Korder-1])]   # tipo minimo
	tipo_p = type[1][type[0].index(morf_p[Korder-1])]   # tipo maximo
	
	etipo_m = tipo - tipo_m  # error INFerior en tipo:  tipo-etipo_m
	etipo_p = tipo_p - tipo  # error SUPerior en tipo:  tipo+etipo_p
	
	return tipos, tipo, tipo_m, tipo_p


