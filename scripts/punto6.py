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


# repuesta escalon

#grafico respuesta al escalon 
sns.set_style('darkgrid')
plt.figure(figsize= (8,6))
t1, h1 = ctrl.step_response(Hnorm)
t2, h2 = ctrl.step_response(Horig)
sns.lineplot(x=1000*t2,y=h2, color='g', linestyle = 'dashdot', label = '$H_{asignada}$', linewidth = 2)
sns.lineplot(x=1000*t1,y=h1, color='r', linestyle = 'dotted', label = '$H_{normalizada}$', linewidth = 1.5)
plt.xlabel('$t\quad [ms]$')
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.title("Respuesta al escalon")
plt.legend(prop={'size': 15})

#respuesta impulso

#grafico respuesta al escalon 
sns.set_style('darkgrid')
plt.figure(figsize= (8,6))
t1, h1 = ctrl.impulse_response(Hnorm)
t2, h2 = ctrl.impulse_response(Horig)
sns.lineplot(x=1000*t2,y=h2, color='g', linestyle = 'dashdot', label = '$H_{asignada}$', linewidth = 2)
sns.lineplot(x=1000*t1,y=h1, color='r', linestyle = 'dotted', label = '$H_{normalizada}$', linewidth = 1.5)
plt.xlabel('$t\quad [ms]$')
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.title("Respuesta al impulso")
plt.legend(prop={'size': 15})

#respuesta seno

t = np.linspace(0, 7e-3, 1000, endpoint=False)
w1 = 1885.169112
u1 = np.sin(w1 * t)
w2 = 11624.40138
u2 = np.sin(w2 * t)
w3 = 21363.633645
u3 = np.sin(w3 * t)
#plt.figure(figsize = (12,8))
#plt.plot(1000*t,  np.sin(2 * np.pi * f * t))
t1, y1, x1 = ctrl.forced_response(Horig, T = t, U = u1, return_x=True)
t11, y11, x11 = ctrl.forced_response(Hnorm, T = t, U = u1, return_x=True)
t2, y2, x2 = ctrl.forced_response(Horig, T = t, U = u2, return_x=True)
t22, y22, x22 = ctrl.forced_response(Horig, T = t, U = u2, return_x=True)
t3, y3, x3 = ctrl.forced_response(Horig, T = t, U = u3, return_x=True)
t33, y33, x33 = ctrl.forced_response(Horig, T = t, U = u3, return_x=True)



fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize = (12,10))

#senial de entrada
w = [w1,w2,w3]
nom = ['01','m','02']
for x in [0,1,2]:
    sns.lineplot(x = 1000*t, y = np.sin(w[x]*t), ax = axs[x], label = "$v_{in}(t)=sen(wt)\quad w = $"+str(w[x]), linewidth = 0.45)
    
# para w01
sns.lineplot(x = 1000*t1, y = y1, ax = axs[0], color = 'g', label = '$H_{asignada}$', linestyle = '-.')
sns.lineplot(x = 1000*t11, y = y11, ax = axs[0], color = 'r', label = '$H_{normalizada}$', linestyle ='dashed',linewidth=.65)
axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 12})
#para wm
sns.lineplot(x = 1000*t2, y = y2, ax = axs[1], color = 'g', label = '$H_{asignada}$')
sns.lineplot(x = 1000*t22, y = y22, ax = axs[1], color = 'r', label = '$H_{normalizada}$', linestyle ='dashed',linewidth=.85)
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 12})
#para w02
sns.lineplot(x = 1000*t3, y = y3, ax = axs[2], color = 'g', label = '$H_{asignada}$')
sns.lineplot(x = 1000*t33, y = y33, ax = axs[2], color = 'r', label = '$H_{normalizada}$', linestyle ='dashed',linewidth=.85)
axs[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 12})


plt.xlabel('$t\quad [ms]$', fontsize = 15)
axs[0].set_ylabel("$v_{out}(t)\quad [V]$")
axs[1].set_ylabel("$v_{out}(t)\quad [V]$")
axs[2].set_ylabel("$v_{out}(t)\quad [V]$")

fig.savefig('respuestaSenoConjunta.png', format='png')

# Respuesta cuadrada

#f0/10

t = np.linspace(0, 10e-3, 1000, endpoint=False)
f = 1850.0809/10
u = signal.square(2 * np.pi * f * t)
sns.set_style('darkgrid')
plt.figure(figsize = (10,8))
plt.plot(1000*t,  signal.square(2 * np.pi * f * t), linewidth= 0.55)
t1, y1, x1 = ctrl.forced_response(Horig, T = t, U = u,return_x=True)
t11, y11, x11 = ctrl.forced_response(Horig, T = t, U = u,return_x=True)
sns.lineplot(x = 1000*t1, y = y1, color = 'g', label = '$H_{asignada}$', linestyle = '-.', linewidth = 1.5)
sns.lineplot(x = 1000*t11, y = y11, color = 'r', label = '$H_{asignada}$', linestyle = 'dashed',linewidth= .95)
plt.xlabel("$t\quad [ms]$")
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.xlim((0,10))
plt.xticks(np.arange(0,11,1))
plt.title("Respuesta a onda cuadrada de frecuencia $f_0/10$")
plt.legend(prop={'size': 15})


#f0

t = np.linspace(0, 10e-3, 1000, endpoint=False)
f = 1850.0809 
u = signal.square(2 * np.pi * f * t)
sns.set_style('darkgrid')
plt.figure(figsize = (10,8))
plt.plot(1000*t,  signal.square(2 * np.pi * f * t), linewidth = .55)
t2, y2, x2 = ctrl.forced_response(Horig, T = t, U = u,return_x=True)
t22, y22, x22 = ctrl.forced_response(Hnorm, T = t, U = u,return_x=True)
sns.lineplot(x = 1000*t2, y = y2, color = 'g', label = '$H_{asignada}$', linestyle = '-.')
sns.lineplot(x = 1000*t22, y = y22, color = 'r', label = '$H_{asignada}$', linestyle = 'dashed',linewidth=.65)
plt.xlabel("$t\quad [ms]$")
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.xlim((0,4))
plt.xticks(np.arange(0,5,1))
plt.title("Respuesta a onda cuadrada de frecuencia $f_0$")


#10f0

t = np.linspace(0, 10e-3, 1000, endpoint=False)
f = 10*1850.0809 
u = signal.square(2 * np.pi * f * t)
sns.set_style('darkgrid')
plt.figure(figsize = (10,8))
plt.plot(1000*t,  signal.square(2 * np.pi * f * t), linewidth = .55)
t3, y3, x3 = ctrl.forced_response(Horig, T = t, U = u,return_x=True)
t33, y33, x33 = ctrl.forced_response(Hnorm, T = t, U = u,return_x=True)
sns.lineplot(x = 1000*t3, y = y3, color = 'g', label = '$H_{asignada}$', linestyle = '-.')
sns.lineplot(x = 1000*t33, y = y33, color = 'r', label = '$H_{asignada}$', linestyle = 'dashed',linewidth=.65)
plt.xlabel("$t\quad [ms]$")
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.xlim((0,2.5))
#plt.xlim((0,0.5))
#plt.xticks(np.arange(0,3,0.5))
#plt.ylim((-0.2,0.2))
plt.title("Respuesta a onda cuadrada de frecuecia $10f_0$")


#bode 

s = ctrl.TransferFunction.s
Hnorm = (453622744.6*s**2)/(s**4 + s**3*32786.3706 + s**2*537594936.7 + s*1.318*10**12 + 1.617*10**15)
Horig = (s**2*(4.564*10**8))/(s**4+s**3*(3.288*10**4) + s**2*(5.405*10**8)+s*(1.324*10**12)+1.622*10**15)

wbode = np.arange(1e+1,1e+6,1e+1)

magOrig, phaseOrig, omegaOrig = ctrl.bode(Horig,wbode, dB = True, deg = True, plot=False)
magNorm, phaseNorm, omegaNorm = ctrl.bode(Hnorm,wbode,dB=True, deg = True, plot = False)

phaseOriginal = []
for x in phaseOrig:
    phaseOriginal.append(x*(180/cmath.pi))
    
phaseNormalizado = []
for x in phaseNorm:
    phaseNormalizado.append(x*(180/cmath.pi))
    
magdBOrig =[]
for x in magOrig:
    magdBOrig.append(20*np.log10(x))

magdBNorm =[]
for x in magNorm:
    magdBNorm.append(20*np.log10(x))



fig, (ax1, ax2) = plt.subplots(2, figsize=(15,9))
plt.figure(figsize= (15,5))
# diagrama de ganancia
sns.lineplot(omegaOrig,magdBOrig, label = "$H_{asignada}$", linestyle = '-.', color = 'g',ax=ax1, linewidth = 1.5)
sns.lineplot(omegaNorm,magdBNorm, label = "$H_{normalizada}$", linestyle = 'dashed', color = 'r', linewidth =.9,ax=ax1)
plt.sca(ax1)
plt.title("Diagrama de Bode de Fase y Magnitud", fontsize = 12)
plt.yticks((np.arange(-65,5,5)))
plt.xlim((1e+1,1e+6))
plt.ylim((-65,5))
plt.xscale("log")
plt.grid(True, which="both", ls="-")
plt.ylabel("$| H(jw) |_{dB}$", fontsize = 12)
plt.legend(prop={'size': 15})

# diagrama de fase 
plt.sca(ax2)
sns.lineplot(omegaOrig,phaseOriginal, label = "$H_{asignada}$", linestyle = '-.', color = 'g',ax=ax2, linewidth = 1.5)
sns.lineplot(omegaNorm,phaseNormalizado, label = "$H_{normalizada}$", linestyle = 'dashed', color = 'r', linewidth =.9,ax=ax2)
plt.xscale("log")
plt.grid(True, which="both", ls="-")
plt.ylabel("$Fase\quad [grados]$", fontsize = 12)
plt.xlabel("$w_{log}\quad (rad/seg)$", fontsize = 12)
plt.legend(prop={'size': 15})
plt.xlim((1e+1,1e+6))
plt.yticks((np.arange(-550,-130,50)))
fig.savefig('diagramaBodeConjunto.png', format='png')