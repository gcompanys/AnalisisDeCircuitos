import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
import seaborn as sns
import cmath
import warnings
sns.set_style('darkgrid')
warnings. filterwarnings("ignore") 


s = ctrl.TransferFunction.s
Hnorm = (453622744.6*s**2)/(s**4 + s**3*32786.3706 + s**2*537594936.7 + s*1.318*10**12 + 1.617*10**15)
Horig = (s**2*(4.564*10**8))/(s**4+s**3*(3.288*10**4) + s**2*(5.405*10**8)+s*(1.324*10**12)+1.622*10**15)


#POLOS Y CEROS 

print("polos",ctrl.pole(Horig))
print("ceros",ctrl.zero(Horig))

# POLE ZERO MAP 

sns.set_style('darkgrid')
p, z = ctrl.pzmap(Horig, title = 'Polos y Ceros de H(s)')
plt.xlabel('σ', fontsize = 14)
plt.ylabel('j w', fontsize = 14)

# RESPUESTA A DIFERENTES EXCITACIONES

#RESPUESTA ESCALON
plt.figure(figsize= (8,6))
t, h = ctrl.step_response(Horig)
sns.lineplot(x=t,y=h, color='g')
plt.xlabel('Tiempo [s]')
plt.ylabel("$v_{out}(t) [V]$")
plt.title("Respuesta al escalon")
#plt.xlim((0,0.005))
#me tira un warning de result may not be accurate pero graficando lo que obtuve analiticamente en desmos
#obtengo el mismo grafico.


#RESPUESTA IMPULSO
plt.figure(figsize=(8,8))
t,h = ctrl.impulse_response(Horig)
sns.lineplot(x=t,y=h, color='g')
plt.xlabel('Tiempo [s]')
plt.ylabel("$v_{out}(t) [V]$")
plt.title("Respuesta al impulso")

# RESPUESTA A SENO ANALITICA

t = np.arange(0,0.005,0.00001)

def v1(x):
    return np.exp(-1332.8325*x)*(-0.2906*np.cos(1333.1991*x)-0.0154*np.sin(1333.1991*x))

def v2(x):
    return np.exp(-15107.1675*x)*(0.7007*np.cos(15105.5729*x)+0.09*np.sin(15105.5729*x))

def v3(x):
    return -0.4101*np.cos(10000*x)+0.8859*np.sin(10000*x)

def respuesta(t):
    return (v1(t) + v2(t) + v3(t))

plt.figure(figsize=(8,8))
sns.lineplot(x = t, y = respuesta(t), color = 'g')
plt.title("Respuesta a sin(wt) con w = 10000rad/s")
plt.xlabel('Tiempo [s]')
plt.ylabel('$V_{out}(t)\quad [V]$')
