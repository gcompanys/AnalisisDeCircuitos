import numpy as np 
import warnings
sns.set_style('darkgrid')
warnings. filterwarnings("ignore") 

s = ctrl.TransferFunction.s
Hnorm = (453622744.6*s**2)/(s**4 + s**3*32786.3706 + s**2*537594936.7 + s*1.318*10**12 + 1.617*10**15)
Horig = (s**2*(4.564*10**8))/(s**4+s**3*(3.288*10**4) + s**2*(5.405*10**8)+s*(1.324*10**12)+1.622*10**15)
H = (s**2*(4.564*10**8))/(s**4+s**3*(3.288*10**4) + s**2*(5.405*10**8)+s*(1.324*10**12)+1.622*10**15)

E96 = [100,102,105,107,110,113,115,118,121,124,127,130,133,137,140,143,147,150,154,158,162,165,169,174,178,182,187,191,
196,200,205,210,215,221,226,232,237,243,249,255,261,267,274,280,287,294,301,309,316,324,332,340,348,357,365,374,383,392,
402,412,422,432,442,453,464,475,487,499,511,523,536,549,562,576,590,604,619,634,649,665,681,698,715,732,750,768,787,
806,825,845,866,887,909,931,953,976]

E24 = [10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91]

posiblesR = []
for x in E96:
    posiblesR.append(x*10)
    posiblesR.append(x*10**2)
    posiblesR.append(x*10**3)
    posiblesR.append(x*10**4)
    
posiblesC = []
for x in E24:
    posiblesC.append(x*10**(-10)) 
    posiblesC.append(x*10**(-9))
    posiblesC.append(x*10**(-8))

#para el pasa bajos

w_ideal = 21363.633645
q_ideal = 0.7071

# x: R1 ---- y: R2

for c1 in posiblesC:
    for c2 in posiblesC:
        for x in posiblesR:
            for y in posiblesR:
                w_norm = 1/(np.sqrt(x*y*c1*c2))
                q_norm = np.sqrt(x*y*c1*c2)*(1/(x*(c1+c2) + y*c2 - x*c1))
                ew = (np.abs(w_ideal - w_norm)/w_ideal)*100
                eq = (np.abs(q_ideal - q_norm)/q_ideal)*100
                if (ew < 1) and (eq < 1):
                    print('R1=',x,'R2=',y,'c1=',c1,'c2=',c2, 'ew=',ew,'eq=',eq)

#para el pasa altos

w_ideal = 1885.169112
q_ideal = 0.7072


# x: R1 ---- y: R2

for c1 in posiblesC:
    for c2 in posiblesC:
        for x in posiblesR:
            for y in posiblesR:
                w_norm = 1/(np.sqrt(x*y*c1*c2))
                q_norm = (w_norm)*((x*c1*c2)/(c1+c2))
                ew = (np.abs(w_ideal - w_norm)/w_ideal)*100
                eq = (np.abs(q_ideal - q_norm)/q_ideal)*100
                if (ew < 1) and (eq < 1):
                    print('R1=',x,'R2=',y,'c1=',c1,'c2=',c2, 'ew=',ew,'eq=',eq)

