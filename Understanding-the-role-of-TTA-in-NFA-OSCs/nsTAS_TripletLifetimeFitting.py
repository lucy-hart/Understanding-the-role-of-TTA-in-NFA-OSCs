#%%
import os
import pandas as pd
import numpy as np 
import scipy as sp
import lmfit
import matplotlib.pyplot as plt
import matplotlib

#%%
#import data
dir = os.getcwd() 
YSeries = 6
Save_output = 1
filename = dir + '\data\Y{}Interp.txt'.format(YSeries)

#for PM6:Y11, polaron signal was 960-970 nm and triplet was 1425-1435 nm 
#for PM6:Y6, polaron signal was 920-940 nm and triplet was 1465-1475 nm 

if YSeries == 6:

    fluences = ['3p8', '7p6', '15', '31', '61']
    titles = ['3.8', '7.6', '15', '31', '61']
    columns = ['time/ps', '3p8', '7p6', '15', '31', '61']

    ar = np.genfromtxt(filename)
    ar[:,1:] *= -1
    sigma = [1e-5, 2e-5, 3e-5, 3e-5, 4e-5]
    
elif YSeries == 11:

    fluences = ['3p2', '6p4', '13', '26', '51']
    titles = ['3.2', '6,4', '13', '26', '51']
    columns = ['time/ps', '3p2', '6p4', '13', '26', '51']

    ar = np.genfromtxt(filename)
    ar[:,1:] *= -1
    sigma = [1e-5, 2e-5, 3e-5, 2e-5, 3e-5]

#Convert time values to seconds
ar[:,0] = ar[:,0]/(1e9)

#Find initial DeltaT/T values
DeltaT_trip0 = np.zeros(len(fluences))
for i in range(len(fluences)):
    DeltaT_trip0[i] = ar[0,i+1]

#%%
#Define the functions used in the program

#Analytic solution for DetltaT_trip
#Assume decay of the form -A*DeltaT - D*(DeltaT)^2
def DeltaT(t, i, params, DeltaT_trip0, t0):
    A = params['A_%i' %i].value
    D = params['D_%i' %i].value
    if A != 0:
        phi = A/(2*D)
        C = ((DeltaT_trip0[i])/(DeltaT_trip0[i] + 2*phi))*np.exp(A*t0)
        DeltaT_trip = phi*((1+C*np.exp(-A*t))/(1-C*np.exp(-A*t))-1)
    elif A == 0:
        DeltaT_trip = 1/(D*(t-t0)+ (1/DeltaT_trip0[i]))
    return DeltaT_trip

#Residuals from fitting the triplet populations   
def objective(params, ar, fluences, DeltaT_trip0, eps):
    tmin = ar[0,0]
    args = [DeltaT_trip0, tmin]
    resid = np.array((ar[:,1] - DeltaT(ar[:,0], 0, params, *args))/eps[0])
    for i in range(len(fluences)-1):
        res = np.array((ar[:,i+2] - DeltaT(ar[:,0], i+1, params, *args))/eps[i+1])
        resid = np.concatenate((resid, res))
    return resid

#%%
#Create parameters for DeltaT_trip
fit_params = lmfit.Parameters()
random = np.random.rand(1)

#Don't use high-fluence data - concern abut sample stability in this measurement
UseAllFluences = 0
if UseAllFluences == 0:
    num = len(fluences) - 1
elif UseAllFluences == 1:
    num = len(fluences)
#Set this to equal one to globally fit parameters to all fluences 
Global_Fit = 1
#Set this to equal one to include both TTA and monoexponential decay in the fits
FitMono = 0

for i in range(num):
    if FitMono == 1:
        fit_params.add('A_%i' %i, 1e5*random[0], min=0, max=1e10)
    elif FitMono == 0:
        fit_params.add('A_%i' %i, 0, vary = False)
    fit_params.add('D_%i' %i, 1e10, min=0, max=1e15)
for i in range(num - 1):
    if Global_Fit == 1:
            fit_params['A_%i' %(i+1)].expr = 'A_0'
            fit_params['D_%i' %(i+1)].expr = 'D_0'

#Do the fitting
mini = lmfit.Minimizer(objective, fit_params, 
                        fcn_args = (ar[:,0:num+1], fluences[0:num], DeltaT_trip0[0:num], sigma[0:num]))
result = mini.minimize() 
lmfit.report_fit(result)

#Save solutions as an array for easy plotting
BestSolution = {}
resid = np.zeros(num)
for i in range(num):
    BestSolution[i] = DeltaT(ar[:,0], i, result.params, DeltaT_trip0, ar[0,0])

#%%
#Plot solutions 
cmap = matplotlib.cm.get_cmap('viridis')
if UseAllFluences == 0:
    colours = [cmap(0.1), cmap(0.3), cmap(0.8), cmap(1.0)]
elif UseAllFluences == 1:
    colours = [cmap(0.1), cmap(0.3), cmap(0,6), cmap(0.8), cmap(1.0)]

fig = plt.figure(figsize=(10,5), facecolor='white')
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

for i in range(num):
    ax.semilogx(ar[:,0]*1e9, 1000*ar[:,i+1], c = colours[i],   
                   label='{} $\mu$J'.format(titles[i]) + ' cm$^{-2}$:')
    ax.semilogx(ar[:,0]*1e9, 1000*BestSolution[i], 
                   c='black', ls='--')
 
ax.set_xlabel('Time (ns)', fontsize=20)
ax.set_ylabel('$|\Delta$T/T$|$ (x10$^{-3}$)', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
if YSeries == 6:
    ax.set_ylim(-0.1, 1.7)
elif YSeries == 11:
    ax.set_ylim(-0.1, 1.9)  
ax.legend(title='Fluence', fontsize=14, title_fontsize = 14)
plt.show() 

#%%
#Save triplet fitting solutions as a .xlsx file
if YSeries == 6:
    if FitMono == 1:
        TripOutput_filename = dir + r'\Results\NeatY6.xlsx'
    elif FitMono == 0:
        TripOutput_filename = dir + r'\Results\NeatY6_TTAOnly.xlsx'
elif YSeries == 11:
    if FitMono == 1:
        TripOutput_filename = dir + r'\Results\NeatY11.xlsx'
    elif FitMono == 0:
        TripOutput_filename = dir + r'\Results\NeatY11_TTAOnly.xlsx'

if Save_output == 1:
    for i in range(num):
        ar2 = np.concatenate(([ar[:,0]*1e9], [BestSolution[i],]), axis = 0)
        ar2 = np.transpose(ar2)
        df = pd.DataFrame(ar2, columns = ['Time/ps', 'DeltaT_trip'])
        if i == 0:
            with pd.ExcelWriter(TripOutput_filename) as writer:
                df.to_excel(writer, sheet_name = fluences[i])
        else:
            with pd.ExcelWriter(TripOutput_filename, mode='a') as writer:
                df.to_excel(writer, sheet_name = fluences[i])


