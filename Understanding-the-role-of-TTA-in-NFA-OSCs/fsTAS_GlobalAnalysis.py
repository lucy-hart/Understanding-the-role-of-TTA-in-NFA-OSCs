#%%
import os
import pandas as pd
import numpy as np
import scipy.integrate 
import lmfit
import matplotlib.pyplot as plt
import matplotlib

#%%
#import data
dir = os.getcwd()
data_dict={}
Use_Y6 = 0
AsCast = 0
Save_output = 0

#for PM6:Y11, polaron signal was 960-970 nm and triplet was 1425-1435 nm 
#for PM6:Y6, polaron signal was 920-940 nm and triplet was 1465-1475 nm 

if Use_Y6 == 1:

    filenameY6 = dir + '\data\PM6Y6.xlsx' 

    #Fitting parameters
    #bounds = limits for fitting the polaron population
    bounds = np.array([[0, 1e-2], [10, 1e5], [10, 1e5]]) 

    fluences = ['1p5', '3p1', '6p3', '11', '15']
    titles = ['1.5', '3.0', '6.3', '11', '15']
    columns = ['time/ps', '1p5', '3p1', '5p1', '7p4', '15']

    data_dict[0] = pd.read_excel(filenameY6 , sheet_name = 'polaron', names = columns, dtype = np.float64)
    data_dict[0][data_dict[0].columns[1:]] = data_dict[0][data_dict[0].columns[1:]]*-1
    data_dict[1] = pd.read_excel(filenameY6 , sheet_name = 'triplet', names= columns, dtype = np.float64)
    data_dict[1][data_dict[1].columns[1:]] = data_dict[1][data_dict[1].columns[1:]]*-1
    #Standard deviations for each fluence, calculated using CalcStd.py
    sigma = [4e-6, 5e-6, 5e-6, 7e-6, 7e-6]
    
elif Use_Y6 == 0:

    if AsCast == 0:
        filenameY11 = dir + '\data\PM6Y11.xlsx'
    elif AsCast == 1:
        filenameY11 = dir + '\data\PM6Y11_ascast.xlsx'

    #Fitting parameters
    #bounds = limits for fitting the polaron population   
    bounds = np.array([[0, 1e-2], [10, 1e5], [10, 1e6]]) 

    if AsCast == 0:
        fluences = ['2p1', '4p2', '8p5', '15', '21']
        titles = ['2.1', '4.2', '8.5', '15', '21']
        columns = ['time/ps', '2p1', '4p2', '8p5', '15', '21']
    elif AsCast == 1:
        fluences = ['2p1', '8p5', '15', '21']
        titles = ['2.1', '8.5', '15', '21']
        columns = ['time/ps', '2p1', '8p5', '15', '21']

    data_dict[0] = pd.read_excel(filenameY11, sheet_name = 'polaron', names = columns, dtype = np.float64)
    data_dict[0][data_dict[0].columns[1:]] = data_dict[0][data_dict[0].columns[1:]]*-1
    data_dict[1] = pd.read_excel(filenameY11, sheet_name = 'triplet', names = columns, dtype = np.float64)
    data_dict[1][data_dict[1].columns[1:]] = data_dict[1][data_dict[1].columns[1:]]*-1
    #Standard deviations for each fluence, calculated using CalcStd.py
    if AsCast == 0:
        sigma = [2e-6, 5e-6, 5e-6, 5e-6, 1e-5]
    elif AsCast == 1:
        sigma = [4e-6, 7e-6, 9e-6, 1e-5]

    
#data_dict is the raw data for plotting e.g., fluence series
#Start from 0.1 ps (100 fs) to avoid any tomfoolery due to the IRF
#Data_dict is for the fitting 
#Start from a time where polarons have stopped being produced
t_start  = 100
Data_dict={}
for i in range(2):
    data_dict[i] = data_dict[i][data_dict[i]['time/ps'] > 0.1]
    Data_dict[i] = data_dict[i][data_dict[i]['time/ps'] >= t_start]

#Save data as an array and convert time values to seconds
array_dict = {}
for i in range(2):
    array_dict[i] = Data_dict[i].to_numpy(copy=True)
    array_dict[i][:,0] = array_dict[i][:,0]/(1e12)

#Find initial DeltaT/T values
DeltaT_trip0 = np.zeros(len(fluences))
for i in range(len(fluences)):
    DeltaT_trip0[i] = array_dict[1][0,i+1]

#%%
#Define the functions used in the program

#Bi-exponenetial fit to polaron DeltaT/T   
#Function defined using t in picseconds in the exponents because I got confused otherwise
def Polaron_DeltaT(t, C, E, tau1, tau2):
    func = C*np.exp(-(t)/tau1) + E*np.exp(-(t)/tau2)
    dev = -C*(np.exp(-(t)/tau1)/tau1)*1e12 - E*(np.exp(-(t)/tau2)/tau2)*1e12
    return func, dev

#Rate equations for the triplet DeltaT
def Triplet_Kinetics(t, DeltaT_trip, A, B, D, C, E, tau1, tau2):
    DeltaT_pol, dDeltaT_dt = Polaron_DeltaT(t*10**12, C, E, tau1, tau2)
    deriv = -A*dDeltaT_dt  - D*DeltaT_trip**2 - B*DeltaT_trip #- B*DeltaT_trip*DeltaT_pol
    return deriv

#Solve Triplet_Kinetics to get DeltaT_triplet(t)    
def ODE_Solution(i, params, C, E, tau1, tau2, DeltaT_trip0, tmin, tmax): 
    sol = scipy.integrate.solve_ivp(
            fun = Triplet_Kinetics,
            method = 'DOP853',
            t_span = (tmin, tmax),
            y0 = [DeltaT_trip0[i]],
            args=(params['A_%i' %i].value, params['B_%i' %i].value, params['D_%i' %i].value, 
                  C[i], E[i], tau1[i], tau2[i]),
            dense_output = True)
    return sol

#Solve Triplet_Kinetics to get DeltaT_triplet(t)  
#Use this for the plots of the residual versus input parameter, not the actual fitting   
def ODE_Solution_ScanResiduals(i, A, B, D, C, E, tau1, tau2, DeltaT_trip0, tmin, tmax): 
    sol = scipy.integrate.solve_ivp(
            fun = Triplet_Kinetics,
            method = 'DOP853',
            t_span = (tmin, tmax),
            y0 = [DeltaT_trip0[i]],
            args=(A, B, D, C[i], E[i], tau1[i], tau2[i]),
            dense_output = True)
    return sol

#Residuals from fitting the polaron populations
def objective1(params, array_dict, fluences):
    resid = np.array(array_dict[0][:,1] - Polaron_DeltaT(array_dict[0][:,0]*1e12, 
                                                      params['C_0'].value, params['E_0'].value, 
                                                      params['tau_0'].value, params['T_0'].value)[0])
    for i in range(len(fluences)-1):
        i+=1
        res = np.array(array_dict[0][:,i+1] - Polaron_DeltaT(array_dict[0][:,0]*1e12, 
                                                      params['C_%i' %i].value, params['E_%i' %i].value, 
                                                      params['tau_%i' %i].value, params['T_%i' %i].value)[0])
        resid = np.concatenate((resid, res))
    return resid

#Residuals from fitting the triplet populations   
def objective2(params, array_dict, fluences, C, E, tau1, tau2, DeltaT_trip0, eps):
    tmin = array_dict[1][0,0]
    tmax = array_dict[1][0,-1]
    args = [C, E, tau1, tau2, DeltaT_trip0, tmin, tmax]
    resid = np.array((array_dict[1][:,1] - ODE_Solution(0, params, *args).sol(array_dict[1][:,0])[0])/eps[0])
    for i in range(len(fluences)-1):
        res = np.array((array_dict[1][:,i+2] - ODE_Solution(i+1, params, *args).sol(array_dict[1][:,0])[0])/eps[i+1])
        resid = np.concatenate((resid, res))
    return resid

#%%
#create the parameter sets for DeltaT_pol
random = np.random.rand(4)                        
fit_params1 = lmfit.Parameters()
for i in range(len(fluences)):
    fit_params1.add('C_%i' %i, bounds[0,1]*random[0], min= bounds[0,0], max=bounds[0,1])
    fit_params1.add('E_%i' %i, bounds[0,1]*random[1], min= bounds[0,0], max=bounds[0,1])
    fit_params1.add('tau_%i' %i, bounds[1,1]*random[2], min= bounds[1,0], max=bounds[1,1])
    fit_params1.add('T_%i' %i, bounds[1,1]*random[3], min= bounds[2,0], max=bounds[2,1])    
                                   
#Do the fitting
#Always check the results of this from the plot (fig) as sometimes it can fit badly, which in turn messes up the triplet fit
result1 = lmfit.minimize(objective1, fit_params1, args = (array_dict, fluences))
C = np.zeros(len(fluences))
tau1 = np.zeros(len(fluences))
tau2 = np.zeros(len(fluences))
E = np.zeros(len(fluences))
for i in range(len(fluences)):
    C[i] = result1.params['C_%i' %i].value
    E[i] = result1.params['E_%i' %i].value
    tau1[i] = result1.params['tau_%i' %i].value
    tau2[i] = result1.params['T_%i' %i].value
print('Polaron Signal Fitted')

#%%
#Save polaron fitting as .xlsx file (for plotting)
if Use_Y6 == 1:
    PolOutput_filename = dir + '\Results\PM6Y6_Polaron.xlsx'
elif Use_Y6 == 0:
    if AsCast == 0:
        PolOutput_filename = dir + '\Results\PM6Y11_Polaron.xlsx'
    elif AsCast == 1:
        PolOutput_filename = dir + '\Results\PM6Y11_ascast_Polaron.xlsx'

if Save_output == 1:
    for i in range(len(fluences)):
        ar1= [Polaron_DeltaT(array_dict[0][:,0]*1e12, C[i], E[i], tau1[i], tau2[i])[0],]
        ar2 = np.concatenate(([array_dict[0][:,0]*1e12,], ar1), axis = 0)
        ar2 = ar2.transpose()
        df = pd.DataFrame(ar2, columns = ['Time/ps', 'DeltaT_pol'])
        if i == 0:
            with pd.ExcelWriter(PolOutput_filename) as writer:
                df.to_excel(writer, sheet_name = fluences[i])
        else:
            with pd.ExcelWriter(PolOutput_filename, mode='a') as writer:
                df.to_excel(writer, sheet_name = fluences[i])

#%%
#create parameter sets for DeltaT_trip
#B is for TCA and D is for TTA
#Choose fitting equation (TTA = 0 => D=0, TTA = 1 => B=0, TTA = 0.5 => both B and D non-zero)
#To fix variables to be the same for all datasets, set Global_Fit = 1
TTA = 0.5
Global_Fit = 1
fit_params2 = lmfit.Parameters()

if TTA == 0:
    for i in range(len(fluences)):
        fit_params2.add('A_%i' %i, 10*random[1], min=0, max=10)
        fit_params2.add('B_%i' %i, 1e13*random[2], min=0, max=1e15) 
        fit_params2.add('D_%i' %i, 0, vary=False)    
    for i in range(len(fluences) - 1):
        fit_params2['A_%i' %(i+1)].expr = 'A_0'
        if Global_Fit == 1:
            fit_params2['B_%i' %(i+1)].expr = 'B_0'
elif TTA == 1:
    for i in range(len(fluences)):
        fit_params2.add('A_%i' %i, 10*random[1], min=0, max=10)
        fit_params2.add('B_%i' %i, 0, vary=False)
        fit_params2.add('D_%i' %i, 1e13*random[3], min=0, max=1e15)    
    for i in range(len(fluences) - 1):
        fit_params2['A_%i' %(i+1)].expr = 'A_0'
        if Global_Fit == 1:
            fit_params2['D_%i' %(i+1)].expr = 'D_0'
elif TTA == 0.5:
    for i in range(len(fluences)):
        fit_params2.add('A_%i' %i, 10*random[1], min=0, max=10)
        fit_params2.add('B_%i' %i, 1e6, min=0, max=1e13) 
        #fit_params2.add('D_%i' %i, 2.3e11, vary=False) 
        fit_params2.add('D_%i' %i, 2.3e11, min=0, max=1e13) 
    for i in range(len(fluences) - 1):
        fit_params2['A_%i' %(i+1)].expr = 'A_0'
        if Global_Fit == 1:
            fit_params2['B_%i' %(i+1)].expr = 'B_0'
            fit_params2['D_%i' %(i+1)].expr = 'D_0'

#Do the fitting
mini = lmfit.Minimizer(objective2, fit_params2, 
                        fcn_args = (array_dict, fluences, C, E, tau1, tau2, DeltaT_trip0, sigma))
result2 = mini.minimize() 
lmfit.report_fit(result2)

#Save solutions as an array for easy plotting
BestSolution = {}
resid = np.zeros(len(fluences))
for i in range(len(fluences)):
    BestSolution[i] = ODE_Solution(i, result2.params, C, E, tau1, tau2, DeltaT_trip0, array_dict[1][0,0], array_dict[1][0,-1]).sol(array_dict[1][:,0])
    resid[i] = np.sqrt(np.sum(((array_dict[1][:,i+1]-BestSolution[i][0]))**2))
print(resid)

#%%
#Plot solutions   
C_values = [result2.params['D_%i' %i].value for i in range(len(fluences))]
cmap = matplotlib.cm.get_cmap('viridis')
colours = [cmap(0.1), cmap(0.4), cmap(0.6), cmap(0.8), cmap(1.0)]
if AsCast == 1:
    colours = [cmap(0.1), cmap(0.5), cmap(0.8), cmap(1.0)]
fig1, ax1 = plt.subplots(1,2, figsize=(10,5), facecolor='white')

for i in range(len(fluences)):
    ax1[0].semilogx(data_dict[0]['time/ps'], data_dict[0][data_dict[0].columns[i+1]]*1e4, c = colours[i], 
                   label='{} $\mu$J'.format(titles[i]) + ' cm$^{-2}$:')
    ax1[0].semilogx(Data_dict[0]['time/ps'], 
                   Polaron_DeltaT(array_dict[0][:,0]*10**12, C[i], E[i], tau1[i], tau2[i])[0]*1e4, 
                   c='black', ls='--')
    ax1[1].semilogx(data_dict[1]['time/ps'], data_dict[1][data_dict[1].columns[i+1]]*1e4, 
                   c = colours[i], label='{} $\mu$J'.format(titles[i]) + ' cm$^{-2}$:')
    ax1[1].semilogx(Data_dict[1]['time/ps'], BestSolution[i][0]*1e4, 
                   c='black', ls='--')

for i in range(2):  
    ax1[i].set_xlabel('time/ps', fontsize=20)
    ax1[i].set_ylabel('$|\Delta$T/T$|$ (x10$^{-4}$)', fontsize=20)
    ax1[i].tick_params(axis='both', labelsize=20)
    ax1[i].set_xlim(90, 2000)
if Use_Y6 == 0:
    ax1[0].set_ylim(0, 11.0)
    ax1[1].set_ylim(0, 13.5)
elif Use_Y6 == 1:
    ax1[0].set_ylim(0, 65.0)
    ax1[1].set_ylim(0, 13.0)   
ax1[0].legend(title='Fluence', fontsize=14, title_fontsize = 14)
plt.tight_layout()

#%%
#Save triplet fitting solutions as a .xlsx file

if (Use_Y6 == 1) and (TTA == 0):
    TripOutput_filename = dir + '\Results\PM6Y6_TCA.xlsx'
elif (Use_Y6 == 1) and (TTA == 1):
    TripOutput_filename = dir + '\Results\PM6Y6_TTA.xlsx'
elif (Use_Y6 == 1) and (TTA == 0.5):
    TripOutput_filename = dir + '\Results\PM6Y6_Both.xlsx'
if (Use_Y6 == 0):
    if AsCast == 0:
        if (TTA == 0):
            TripOutput_filename = dir + '\Results\PM6Y11_TCA.xlsx'
        elif (TTA == 1):
            TripOutput_filename = dir + '\Results\PM6Y11_TTA.xlsx'
        elif  (TTA == 0.5):
            TripOutput_filename = dir + '\Results\PM6Y11_Both.xlsx'
    elif AsCast == 1:
        TripOutput_filename = dir + '\Results\PM6Y11_ascast_TTA.xlsx'

if Save_output == 1:
    for i in range(len(fluences)):
        ar3 = np.concatenate(([array_dict[1][:,0]*1e12,], [BestSolution[i][0],]), axis = 0)
        ar3 = np.transpose(ar3)
        df = pd.DataFrame(ar3, columns = ['Time/ps', 'DeltaT_trip'])
        if i == 0:
            with pd.ExcelWriter(TripOutput_filename) as writer:
                df.to_excel(writer, sheet_name = fluences[i])
        else:
            with pd.ExcelWriter(TripOutput_filename, mode='a') as writer:
                df.to_excel(writer, sheet_name = fluences[i])
   


