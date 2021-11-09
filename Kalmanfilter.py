import csv
import numpy as np #using numpy as np
import matplotlib.pyplot as plt# using matplotlib.pyplot as plt

dt = 0.1
A = np.array([[1.0, dt],[0.0, 1.0]]) #2D array
B = np.array([[0.0],[1.0]])#2D array
C = np.array([[1.0, 0.0]]) # 2D array

dataFile = 'csv/data.csv' #Option: data.csv (break after sometimes), data1.csv (break then accelerate), data2.csv (no maneuver)

# making data frame 
#%matplotlib inline

time, x, xdot, xmeasured= [], [], [], []

with open(dataFile, 'r') as file:
    reader = csv.reader(file, delimiter=',') #delimiter: how the data is seperated
    for row in reader:
        time.append(float(row[0]))
        x.append(float(row[1]))
        xdot.append(float(row[2]))
        xmeasured.append(float(row[3]))
        
# list -> numpy array        
time = np.array(time)
x = np.array(x)   
xdot = np.array(xdot)
xmeasured = np.array(xmeasured)  

plt.plot(time,x, label = 'True x',c = 'b')   
plt.plot(time,xdot, label = 'xdot',c = 'g')   

plt.plot(time,xmeasured, 'r+',label = 'x measured')
plt.legend()
plt.xlabel('time')
plt.ylabel('x')

def kalmanPrediction(x,P):
    xPred = A@x
    PPred = A@P@A.T +Q
    return xPred, PPred

#Aufgabe 2: Prozess- und Messrauschen hier ändern
q0 = 0.001
Q = np.array ([[1/3*dt**3,1/2*dt**2],[1/2*dt**2,dt]]) * q0 # Prozessrauschen. 2D np.array
R = 0.01  # Messrauschen 
''' Python Numpy: 
        matrix multiplication: np.matmul(a,b) or a@b
        element-wised operation: a*b, a/b, a+b, a-b
    Matlab:
        matrix multiplication: a*b
        element-wised operation: a.*b
'''
def output(x):
    z = C @ x 
    return z #(1x2) mul (2x1)

def runKalmanFilter (A,B,C,Q,R,xmeasured):
    PPred =np.array( [[0.1, 0],[0, 0.1]] )# Initial first estimate
        
    xEst= np.empty([2, len(xmeasured)]) # init 
    xEst[0][0]= 0 # init
    xEst[1][0]= 1 # init
    for i in range(1,len(xmeasured),1):
        # PREDICTION  
        xPred, PPred = kalmanPrediction(xEst[:,i-1:i],PPred) # (2,1)
        
        #UPDATE
        
        K = PPred@C.T * (1/(float(C @PPred @C.T) + R)) #Kalman gain 
        y = xmeasured[i] - output(xPred) #Error
       
        xEst[:,i:i+1] =  xPred + (K*y)    #compute estimate
        
        PPred = (np.eye(len(K)) - K @C) @ PPred # update error covariance

    return xEst

xest = runKalmanFilter(A,B,C,Q,R,xmeasured)          #runKalmanfilter

plt.plot (time, xest[0],label = 'xEst KF',c ='cyan') #plot  xEst
plt.plot(time,x,label = 'True X',c = 'blue')         #plot  xTrue
plt.plot(time,xmeasured, 'r+',label = 'x measured')  #plot  xMeasured
#plt.plot (time, xest[1],label = 'Xdot',marker="x",c ='orange' )

plt.xlabel('time')
plt.ylabel('x')
plt.legend()

plt.show()