#%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
DT= 0.1
SIMTIME  = 30
total_time_step = int(SIMTIME/DT)

a11= 1
a12= DT
a21= 0
a22= 1

'''
    Aufgabe 1: Matrix A ausfüllen.    
    Hinweis:
        Die A-Matrix wie in der Beschreibung oben.
        a33, a34, a43, a44  wie Variablen zuweisen.
        4 Lines of Code
        löschen "raise NotImplementedError()", wenn Sie mit Ihren Task fertig sind
'''
a33= 1
a34= DT
a43= 0
a44= 1


A = np.array ([[a11 ,a12, 0, 0],
               [a21,a22, 0, 0],
               [0, 0, a33, a34],
               [0, 0, a43, a44]]) #useD by Kalman but not by dataGeneration
               
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])


q0 = 0.001

Q = np.array ([[1/3*DT**3,0.5*DT**2, 0 ,0],
               [0.5*DT**2,DT,0 ,0 ],
               [0, 0, 1/3*DT**3,0.5*DT**2],
               [0, 0, 0.5*DT**2,DT]]) * q0 # Prozessrauschen. 2D np.array
R = np.array([[0.05, 0],
              [0 ,0.05]]) # Messrauschen, 2Dndarray

DT = 0.1

#init zustandvector, messungvektor, Steuervektor.
x  = np.empty((4,total_time_step)) #shape (4,total time)
u = np.empty((2,total_time_step))
xm = np.empty((2,total_time_step))

#interne Datenerfassung. Bitte ändern Sie diesen Block/Funtion nicht
def dataGeneration():
    sigv = 0.1
    sigphi = 0.2
    sigM = 0.2
    np.random.seed(3)
    #init
    u[0,0] = 0
    u[1,0] = 1
    
    x[0,0] = 0.1
    x[1,0] = np.cos( u[0,0])* u[1,0]
    x[2,0] = 0.1
    x[3,0] = np.sin( u[0,0])* u[1,0]
  
    
    for i in range(1,total_time_step,1):
        #internal state here!! for data generations
        
        #v and phi are exicited by noise
        u[0,i] =  u[0,i-1]  + np.random.randn()*sigphi
        u[1,i] =  u[0,i-1] +  np.random.randn()*sigv
        
        
        x[0,i] = x[0,i-1] + DT*x[1,i-1] #  x[0][i] =  x(0,i)
        x[1,i] = np.cos(u[0,i])*u[1,i]
        x[2,i] = x[2,i-1]+ DT*x[3,i-1]
        x[3,i] = np.sin(u[0,i])*u[1,i]
       
        
        xm[0,i] = x[0,i] + sigM*np.random.randn()    
        xm[1,i] = x[2,i] + sigM*np.random.randn()
        
    return x, xm

def motion_modell(x):
    x= A@x #np.mathmul()
    return x 

def observation_modell(x):
    z = C @ x 
    return z

def output(x):
    z = C @ x #np.mathmul()
    return z 

def kalmanPrediction(x,P): #vorherige 1D-Version KF Übung
    xPred= motion_modell (x)
    PPred = A@P@A.T + Q
    return xPred, PPred

def kalmanUpdate(xPred,PPred, i): 
    '''
    Aufgabe 2: kalmanUpdate, Ein Kalman-Korrektur durchführen. Diese Funktion wird dann in der Schleife ausgeführt.
    Eingabe: 
        xPred: bisherige x Prädiktion
        PPred: bisherige P Prädiktion
        i: Index zum Zugriff auf die aktuelle Messung.
    Ausgabe:
        xEst: nächste X KF-Schätzung
        PPred: nächste P KF-Schätzung
    Hinweis:
        siehen vorherige 1D-Version KF.
        C,R: global Variable
        np.linalg.inv(): 
        output() Funktion oben.
        np.expand_dims(): benötigt für xmeasured[:,i]. 
        @ oder np.matmul():
        5-6 Lines of Code
    '''
    
    # YOUR CODE HERE
    K = PPred@C.T@np.linalg.inv(C@PPred@ C.T + R) # */ @ math mul : matlab: .*
    y = np.expand_dims (xmeasured[:,i], axis = 1) - output (xPred) # different between measurement and model
    xEst = xPred + (K@y)
    PPred = (np.eye(len(K)) - K@C) @ PPred # Update Covariance matrix 
    
    
    
    K = PPred@C.T@ np.linalg.inv(C@PPred@ C.T + R)
    y = np.expand_dims(xmeasured[:,i],axis=1)-output(xPred)#Error
    xEst=xPred+(K@y)
    PPred=(np.eye(len(K))-K@C)@PPred

    
    return xEst, PPred
    

def runKalmanFilter (A,C,Q,R,xmeasured):
    PPred =np.array( [[0.1, 0,0,0],
                      [0,0, 0.1,0],
                      [0, 0,0.1,0],
                      [0, 0,0,0.1],] )#Erste Schätzung 4x4
   
        
    xEst= np.empty([4, len(xmeasured[1])]) # init 
    xEst[0,0]= 0 # init
    xEst[1,0]= 1 # init
    xEst[2,0]= 0 # init
    xEst[3,0]= 1 # init
    
    for i in range(1,len(xmeasured[1]),1):
        # PREDICTION  
        xPred, PPred = kalmanPrediction(xEst[:,i-1:i],PPred) # (2,1)
      
        #UPDATE
        xEst[:,i:i+1], PPred = kalmanUpdate(xPred,PPred, i)
        
    return xEst

x, xmeasured = dataGeneration() # x:zustände. xmeasured: GPS Messungen.
xest = runKalmanFilter(A,C,Q,R,xmeasured)  

plt.plot(x[0,:], x[2,:], 'b+',label = 'TrueX',c = 'blue')         #plot  x,y
plt.plot(xmeasured[0,:], xmeasured[1,:], 'r+',label = 'xMessung')  #plot  xMeasured
plt.plot (xest[0,:], xest[2,:],label = 'KFGeschätzte',c ='orange')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
