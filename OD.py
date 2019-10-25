#Method of Gauss
#Lauren Fossel

from __future__ import division
from math import *
import numpy as np
from FosselGaussFunctions import *

#Constants
k = .01720209895
mu = 1 #Gaussian units
c = 173.144632674 #In AU/Julian day
wantSeries = True #True to calculate using f and g series, false to use functions

#Parse through input file:
data = np.genfromtxt("FosselInput.txt", dtype = None)

row1 = list(data[0])
rowN = list(data[-1])

time1 = np.array(row1[3].split(":")).astype(np.float) #Hr, m, sec as elements in a numpy array
time3 = np.array(rowN[3].split(":")).astype(np.float)

#First observation time, ra, dec
data1 = np.array([calcJulian(row1[0], row1[1], row1[2], time1[0], time1[1], time1[2]),
                 convertRa(row1[4], row1[5], row1[6]),
                 convertDec(row1[7], row1[8], row1[9])])

#Last observation time, ra, dec
data3 = np.array([calcJulian(rowN[0], rowN[1], rowN[2], time3[0], time3[1], time3[2]),
                 convertRa(rowN[4], rowN[5], rowN[6]),
                 convertDec(rowN[7], rowN[8], rowN[9])])

#R 1 and 3 vectors
Rvec1 = np.array([row1[10], row1[11], row1[12]]) #In equatorial
Rvec3 = np.array([rowN[10], rowN[11], rowN[12]])

middleObsInx = 1

for i in range(len(data)-2):

    print ("Information for days 1, ", (middleObsInx+1), ", ", len(data), ":\n")
    
    #Middle observation info
    row2 = list(data[middleObsInx])
    time2 = np.array(row2[3].split(":")).astype(np.float)
    
    data2 = np.array([calcJulian(row2[0], row2[1], row2[2], time2[0], time2[1], time2[2]),
                 convertRa(row2[4], row2[5], row2[6]),
                 convertDec(row2[7], row2[8], row2[9])])
    
    Rvec2 = np.array([row2[10], row2[11], row2[12]])

    #Calculate rho
    rhoHatArr = np.array([[cos(data1[1])*cos(data1[2]), sin(data1[1])*cos(data1[2]), sin(data1[2])], #Obs 1, Rho hat 1
                        [cos(data2[1])*cos(data2[2]), sin(data2[1])*cos(data2[2]), sin(data2[2])], #Obs 2, Rho hat 2
                        [cos(data3[1])*cos(data3[2]), sin(data3[1])*cos(data3[2]), sin(data3[2])]]) #Obs 3, Rho hat 3

    #Calculate D values
    D1j = np.array([np.cross(Rvec1, rhoHatArr[1]).dot(rhoHatArr[2]), #D11
                 np.cross(Rvec2, rhoHatArr[1]).dot(rhoHatArr[2]), #D12
                 np.cross(Rvec3, rhoHatArr[1]).dot(rhoHatArr[2])]) #D13

    D2j = np.array([np.cross(rhoHatArr[0], Rvec1).dot(rhoHatArr[2]), #D21
                 np.cross(rhoHatArr[0], Rvec2).dot(rhoHatArr[2]), #D22
                 np.cross(rhoHatArr[0], Rvec3).dot(rhoHatArr[2])]) #D23

    D3j = np.array([rhoHatArr[0].dot(np.cross(rhoHatArr[1], Rvec1)), #D31
                 rhoHatArr[0].dot(np.cross(rhoHatArr[1], Rvec2)), #D32
                 rhoHatArr[0].dot(np.cross(rhoHatArr[1], Rvec3))]) #D33

    #Calculate Taus
    Tau = calcTau(data3[0], data1[0])
    Tau1 = calcTau1(data1[0], data2[0])
    Tau3 = calcTau3(data3[0], data2[0])

    #Calculate A1, B1, A3, B3
    A1 = Tau3/Tau
    B1 = (A1/6)*(Tau**2-Tau3**2)
    A3 = -Tau1/Tau
    B3 = (A3/6)*(Tau**2-Tau1**2)

    #Calculate E and F
    E = -2*(rhoHatArr[1].dot(Rvec2))
    F = Rvec2.dot(Rvec2)

    #Calculate A, B, and Dnot
    Dnot = rhoHatArr[0].dot(np.cross(rhoHatArr[1], rhoHatArr[2]))
    A = (A1*D2j[0]-D2j[1]+A3*D2j[2])/(-1*Dnot)
    B = (B1*D2j[0]+B3*D2j[2])/(-1*Dnot)
                          
    #Calculate a, b, c coefficients
    a_ = -(A**2+A*E+F)
    b_ = -mu*(2*A*B+B*E)
    c_ = -mu**2*B**2

    #Calculate roots of r
    rootsArr = np.roots(np.array([1,0,a_,0,0,b_,0,0,c_]))
    rootsArr=rootsArr[np.isreal(rootsArr)]
    finalRootsList = []
    finalRootsListReal = []
    
    for element in rootsArr:
        if element>0 and np.isreal(element):
            finalRootsList.append(element)

    for element in finalRootsList:
        finalRootsListReal.append(np.linalg.norm(element))
        
    finalRootsListReal = np.asarray(finalRootsListReal)
    if len(finalRootsListReal)>1:
        count = 1
        for element in finalRootsListReal:
            print ("Root #", count, ": ", element)
            count+=1
        choice = input("Please enter the number of the root you want to use: ")
        print("")
        if choice==1:
            r2 = finalRootsListReal[0]
        elif choice==2:
            r2 = finalRootsListReal[1]
        elif choice==3:
            r2 = finalRootsListReal[2]
    else:
        r2 = finalRootsListReal[0]

    #Find first values of f1, f3, g1, g3
    f1 = 1-(1/(2*r2**3))*Tau1**2
    f3 = 1-(1/(2*r2**3))*Tau3**2
    g1 = Tau1-(1/(6*r2**3))*Tau1**3
    g3 = Tau3-(1/(6*r2**3))*Tau3**3

    #Estimage rhos
    rho1, rho2, rho3 = calcRho(f1, f3, g1, g3, D1j, D2j, D3j, Dnot)

    #Estimate rho vectors:
    rhoVec1, rhoVec2, rhoVec3 = calcRhoVec(rho1, rho2, rho3, rhoHatArr)

    #Estimate rVec:
    rVec1, rVec2, rVec3 = calcrVec(rhoVec1, rhoVec2, rhoVec3, Rvec1, Rvec2, Rvec3)

    #Estimate calcr2VecDot:
    r2VecDot = calcr2VecDot(rVec1, rVec3, f1, f3, g1, g3)
    
    rVec2prev = np.zeros(3)
    count = 0
    
    #Iterate
    while (np.linalg.norm(rVec2-rVec2prev)>1e-12):

        rVec2prev = rVec2

        #Correct for light travel time
        t1 = data1[0]-rho1/c
        t2 = data2[0]-rho2/c
        t3 = data3[0]-rho3/c

        #Calcualte new Taus
        Tau1 = calcTau1(t1, t2)
        Tau = calcTau(t3, t1)
        Tau3 = calcTau3(t3, t2)

        #Calculate a and n
        a = calcBabyOD(rVec2, r2VecDot)[0]
        n = sqrt(mu/a**3)

        if wantSeries:
            #Calculate series for f and g 
            u = mu/np.linalg.norm(rVec2)**3
            z = (rVec2.dot(r2VecDot))/np.linalg.norm(rVec2)**2
            q = (r2VecDot.dot(r2VecDot))/np.linalg.norm(rVec2)**2-u
        
            f1 = 1-.5*u*Tau1**2+.5*u*z*Tau1**3+(1/24)*(3*u*q-15*u*z**2+u**2)*Tau1**4
            f3 = 1-.5*u*Tau3**2+.5*u*z*Tau3**3+(1/24)*(3*u*q-15*u*z**2+u**2)*Tau3**4

            g1 = Tau1-(1/6)*u*Tau1**3+.25*u*z*Tau1**4
            g3 = Tau3-(1/6)*u*Tau3**3+.25*u*z*Tau3**4

        else:
            #Calculate delta Es
            E1 = deltaENewtonMethod(a, Tau1, rVec2, r2VecDot)
            E3 = deltaENewtonMethod(a, Tau3, rVec2, r2VecDot)

            #Calculate new f1, f3, g1, g3 using functions:
            f1 = 1-(a/np.linalg.norm(rVec2))*(1-cos(E1))
            f3 = 1-(a/np.linalg.norm(rVec2))*(1-cos(E3))
            g1 = k*(t1-t2)+(1/n)*(sin(E1)-E1)
            g3 = k*(t3-t2)+(1/n)*(sin(E3)-E3)

        #Calculate rho mags
        rho1, rho2, rho3 = calcRho(f1, f3, g1, g3, D1j, D2j, D3j, Dnot)

        #Calculate rho vectors
        rhoVec1, rhoVec2, rhoVec3 = calcRhoVec(rho1, rho2, rho3, rhoHatArr)
        
        #Calculate new rVec
        rVec1, rVec2, rVec3 = calcrVec(rhoVec1, rhoVec2, rhoVec3, Rvec1, Rvec2, Rvec3)

        #Calculate new r2VecDot
        r2VecDot = calcr2VecDot(rVec1, rVec3, f1, f3, g1, g3)
        count+=1

    obliq = radians(23.43701)
    a, e, i, Om, w, M = calcBabyOD(rVec2, r2VecDot)
    print (degrees(M), " M")

    #Convert elements to ecliptic
    iprime = acos(cos(obliq)*cos(i)+sin(obliq)*sin(i)*cos(Om))
    Omprime = atan2(sin(Om)*(sin(i)/sin(iprime)), (cos(obliq)*cos(iprime)-cos(i))/(sin(obliq)*sin(iprime)))
    wMinuswprime = atan2(sin(obliq)*(sin(Om)/sin(iprime)), cos(Om)*cos(Omprime)+sin(Om)*sin(Omprime)*cos(obliq))

    #Convert vectors to ecliptic
    rVec2Ecl = np.array([rVec2[0], rVec2[1]*cos(obliq)+rVec2[2]*sin(obliq), -1*rVec2[1]*sin(obliq)+rVec2[2]*cos(obliq)])
    r2VecDotEcl = np.array([r2VecDot[0], r2VecDot[1]*cos(obliq)+r2VecDot[2]*sin(obliq), -1*r2VecDot[1]*sin(obliq)+r2VecDot[2]*cos(obliq)])
    r2VecDotEcl = r2VecDotEcl*k
            
    #Precess M
    mu_ = k**2
    n_ = sqrt(mu_/a**3)
    tnot = calcJulian(2017, 7, 22, 6, 0, 0)
    Mnew = M + n_*(tnot-t2)

    print ("a: ", round(a,4), " AU")
    print ("e: ", round(e,4))
    print ("i: ", round(degrees(iprime),4), " degrees")
    print ("O: ", round(degrees(Omprime)%360,4), " degrees")
    print ("w: ", round(degrees(w-wMinuswprime)%360,4), " degrees")
    print ("M: ", round(degrees(Mnew)%360,4), " degrees")
    print ("Position vector (AU): ", rVec2Ecl)
    print ("Velocity vector (AU/day): ", r2VecDotEcl)
    print ("Range to asteroid: ", round(rho2,4), " AU\n")
    print ("-----------------------------------------------------------------------")

    middleObsInx+=1
