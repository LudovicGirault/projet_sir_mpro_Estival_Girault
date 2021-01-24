import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.integrate import solve_ivp

## Etat initial 
E_init = np.zeros(3)
N = 70000
E_init[1] = 1
E_init[0] = N-E_init[1]


#beta = 1
#gamma = 0.4

#beta = 1.2
#gamma = 0.2


def DureeSejour(etat,beta,gamma,N):
	tps_guerison = -np.log(np.random.rand())/(gamma*etat[1])
	tps_conta = -np.log(np.random.rand())/(beta*etat[0]*etat[1]/N)
	if tps_guerison < tps_conta :
		return tps_guerison, 1
	else :
		return tps_conta, 2

def NouvelEtat(etat,status) :
	if status==1 :
		etat[1] = etat[1] - 1
		etat[2] = etat[2] + 1
	else :
		etat[0] = etat[0] - 1
		etat[1] = etat[1] + 1
	return etat


def SIR(T,N,beta,gamma,E_init):
	etat = E_init
	trajectoire = [etat]
	temps = 0
	sauts = [0]
	while temps<T and etat[0]>0 and etat[1]>0:
		tps, status = DureeSejour(etat,beta,gamma,N)
		etat = NouvelEtat(etat,status)
		trajectoire += [copy.deepcopy(etat)]
		temps += tps
		sauts += [temps]
	return sauts,trajectoire

#x,y = SIR(500,N,beta,gamma,E_init)
#plt.plot(x,y,drawstyle='steps-post')
#plt.show()



#### Methode deterministe

def systeme(t, Y):
	s = Y[0]
	i = Y[1]
	r = Y[2]
	beta = 1.2
	gamma = 0.2
  
	ds_dt = -beta*s*i
	di_dt = beta*s*i - gamma*i
	dr_dt = gamma*i
	
	return [ds_dt,di_dt,dr_dt]

solution = solve_ivp(systeme,[0,50],E_init/N,max_step=0.1)
s = solution.y[0]
i = solution.y[1]
r = solution.y[2]
plt.plot(solution.t,solution.y[0]*N,drawstyle='steps-post')
plt.plot(solution.t,solution.y[1]*N,drawstyle='steps-post')
plt.plot(solution.t,solution.y[2]*N,drawstyle='steps-post')
plt.show()
