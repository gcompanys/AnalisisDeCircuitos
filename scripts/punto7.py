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


#tener los .txt en el mismo directorio

def getTxt(nombre):
    ts,ys = [], []
    with open(nombre,"r") as archivo:
        for linea in archivo:
            row = linea.split()
            x = float(row[0].split(",")[0])
            y = float(row[0].split(",")[1])
            ts.append(x)
            ys.append(y)
    return ts,ys

# respuesta escalon
#grafico respuesta al escalon 
ts, ys = getTxt("respuestaEscalon.txt")
sns.set_style('darkgrid')
plt.figure(figsize= (8,6))
t1, h1 = ctrl.step_response(Hnorm)
t2, h2 = ctrl.step_response(Horig)
sns.lineplot(x=ts,y=ys, color='orange', linestyle = 'dashed', label = 'Tension simulada', linewidth = 2)
sns.lineplot(x=t2,y=h2, color='g', linestyle = '-.', label = 'Tension ideal', linewidth = 2)
sns.lineplot(x=t1,y=h1, color='r', linestyle = 'dotted', label = 'Tension normalizada', linewidth = 2)
plt.xlabel('$t\quad [ms]$')
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.title("Respuesta al escalon")
plt.legend(prop={'size': 15})
plt.xlim((-0.0001,0.005))


#respuesta seno
tseno, yseno = getTxt("respuestaSeno.txt")
plt.figure(figsize=(8,6))
t = np.linspace(0, 7e-3, 1000, endpoint=False)
u = np.sin(2*cmath.pi*1850.0809*t)
#plt.figure(figsize = (12,8))
#plt.plot(1000*t,  np.sin(2 * np.pi * f * t))
t1, y1, x1 = ctrl.forced_response(Horig, T = t, U = u, return_x=True)
t2, y2, x1 = ctrl.forced_response(Hnorm, T = t, U = u, return_x=True)
sns.lineplot(x = tseno, y = yseno, linestyle = 'dashdot', color ='orange', linewidth = 2, label ="Tension simulada")
sns.lineplot(x = t1, y = y1, linestyle = '-.', color ='g', linewidth = 2, label = "Tension ideal")
sns.lineplot(x = t2, y = y2, linestyle = 'dotted', color ='r', linewidth = 2, label = "Tension normalizada")
plt.xlim(0,0.005)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 12})
plt.xlabel("$t\quad [s]$")
plt.ylabel("$v_{out}(t)\quad [V]$")


#respuesta cuadrada

tcua, ycua = getTxt("respuestaCuadrada.txt")
t = np.linspace(0, 10e-3, 1000, endpoint=False)
f = 1850.0809 
u = signal.square(2 * np.pi * f * t)
sns.set_style('darkgrid')
plt.figure(figsize = (10,8))
plt.plot(t,  signal.square(2 * np.pi * f * t), linewidth = .55)
t2, y2, x2 = ctrl.forced_response(Horig, T = t, U = u,return_x=True)
t22, y22, x22 = ctrl.forced_response(Hnorm, T = t, U = u,return_x=True)
sns.lineplot(x = tcua, y = ycua, color = 'orange', label = 'Tension simulada', linestyle = 'dashdot',linewidth=2.5)
sns.lineplot(x = t2, y = y2, color = 'g', label = 'Tension ideal', linestyle = '-.',linewidth= 1.5)
sns.lineplot(x = t22, y = y22, color = 'r', label = 'Tension normalizada', linestyle = 'dotted',linewidth=1.5)
plt.xlabel("$t\quad [s]$")
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.xlim((0,0.004))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 12})




#bode 

omegaBode, magdB , phase = [],[],[]
with open("bode.txt","r") as archivo:
        for linea in archivo:
            row = linea.split()
            x = float(row[0].split(",")[0])
            y = float(row[0].split(",")[1])
            z = float(row[0].split(",")[2])
            omegaBode.append(x)
            magdB.append(y)
            phase.append(z)
            
            

wbode = np.arange(1e+1,1e+6,1e+1)

magOrig, phaseOrig, omegaOrig = ctrl.bode(Horig,wbode, dB = True, deg = True, plot=False)
magNorm, phaseNorm, omegaNorm = ctrl.bode(Hnorm,wbode,dB=True, deg = True, plot = False)

phaseOriginal = []
for x in phaseOrig:
    phaseOriginal.append(x*(180/cmath.pi) + 360)
    
phaseNormalizado = []
for x in phaseNorm:
    phaseNormalizado.append(x*(180/cmath.pi) + 360)
    
magdBOrig =[]
for x in magOrig:
    magdBOrig.append(20*np.log10(x))

#tengo que reescalar los eje de abcisas porque ltspice grafica en Hz
omegaBodeFinal  = []  
for x in omegaBode:
    omegaBodeFinal.append(x*10**(np.log10(2*cmath.pi)))


#magnitud
plt.figure(figsize=(10,6))
sns.lineplot(x = omegaBodeFinal, y = magdB, color = 'orange', linestyle = 'dashdot', linewidth = 3.5, label = "Simulado")
sns.lineplot(x = omegaOrig, y = magdBOrig, color = 'g', linestyle = '-.', linewidth = 2.5, label = "Ideal")
sns.lineplot(x = omegaNorm, y = magdBNorm, color = 'r', linestyle = 'dotted', linewidth = 2.5, label = "Normalizado")
plt.xscale("log")
plt.xlim((10,5e+5))
plt.ylim((-50,1))
plt.ylabel("$| H(jw) |_{dB}$", fontsize = 12)
plt.xlabel("$w_{log}$")
plt.grid(True, which="both", ls="-")

#fase
plt.figure(figsize=(10,6))
sns.lineplot(x = omegaBodeFinal, y = phase, color = 'orange', linestyle = 'dashdot', linewidth = 2.5, label = "Simulado")
sns.lineplot(x = omegaOrig, y = phaseOriginal, color = 'g', linestyle = '-.', linewidth = 2.5, label = "Ideal")
sns.lineplot(x = omegaNorm, y = phaseNormalizado, color = 'r', linestyle = 'dotted', linewidth = 2.5, label = "Normalizado")
plt.xscale("log")
plt.xlim((0,1e+5))
plt.grid(True, which="both", ls="-")
#plt.ylim((-50,1))
plt.xlabel("$w_{log}$")
plt.ylabel("Fase [grados]")


#respuesta para cuadrada 10f0 con simulacion
tc,yc = getTxt("cuadrada10f0.txt")

#cambio de signo dado que arrancaron en distinto nivel la simulacion en ltspice y en python
y_c = []
for x in yc:
    y_c.append(x*(-1))


t = np.linspace(0, 10e-3, 1000, endpoint=False)
f = 1850.0809 
u = signal.square(2 * np.pi * 10* f * t)
sns.set_style('darkgrid')
plt.figure(figsize = (10,8))
plt.plot(t,  signal.square(2 * np.pi *10* f * t), linewidth = .55)
t2, y2, x2 = ctrl.forced_response(Horig, T = t, U = u,return_x=True)
t22, y22, x22 = ctrl.forced_response(Hnorm, T = t, U = u,return_x=True)
sns.lineplot(x = tc, y = y_c, color = 'orange', label = 'Tension simulada', linestyle = 'dashdot',linewidth=2.5)
sns.lineplot(x = t2, y = y2, color = 'g', label = 'Tension ideal', linestyle = '-.',linewidth= 1.5)
sns.lineplot(x = t22, y = y22, color = 'r', label = 'Tension normalizada', linestyle = 'dotted',linewidth=1.5)
plt.xlabel("$t\quad [s]$")
plt.ylabel("$v_{out}(t)\quad [V]$")
plt.xlim((0,0.0005))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 12})
