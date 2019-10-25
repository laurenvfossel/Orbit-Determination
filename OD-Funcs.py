#Lauren Fossel
#Functions for method of Gauss
from __future__ import division
from math import *
import numpy as np

k = .01720209895
mu = 1
c =  173.144632674 #AU/Julian day

#Functions to calculate Taus
def calcTau(t3, t1):
    return k*(t3-t1)
def calcTau1(t1, t2):
    return k*(t1-t2)
def calcTau3(t3,t2):
    return k*(t3-t2)

#Converts Ra to radians
def convertRa(hr, m, sec):
    return radians(15*(hr+m/60+sec/3600))

#Converts Dec to radians
def convertDec(deg, m, sec):
    if deg<0:
        return radians(-1*(abs(deg)+m/60+sec/3600))
    else:
        return radians(abs(deg)+m/60+sec/3600)

#Calculates Julian Day
def calcJulian(yr, mo, d, hr, mi, sec): #Time in UT
    utTime = hr+mi/60+sec/3600
    utFraction = utTime/24
    jNot = 367*yr-int((7*(yr+int((mo+9)/12)))/4)+int((275*mo)/9)+d+1721013.5
    jd = jNot+utFraction
    return jd

#Calculate rho1, rho2, rho3
def calcRho(f1, f3, g1, g3, D1j, D2j, D3j, Dnot):

    #Calculate c1, c2, c3
    c1 = g3/(f1*g3-g1*f3)
    c2 = -1
    c3 = -g1/(f1*g3-g1*f3)
    
    rho1 = (c1*D1j[0]+c2*D1j[1]+c3*D1j[2])/(c1*Dnot)
    rho2 = (c1*D2j[0]+c2*D2j[1]+c3*D2j[2])/(c2*Dnot)
    rho3 = (c1*D3j[0]+c2*D3j[1]+c3*D3j[2])/(c3*Dnot)

    return rho1, rho2, rho3

#Calculate rho vectors
def calcRhoVec(rho1, rho2, rho3, rhoHatArr):
    rhoVec1 = rho1*rhoHatArr[0]
    rhoVec2 = rho2*rhoHatArr[1]
    rhoVec3 = rho3*rhoHatArr[2]
    return rhoVec1, rhoVec2, rhoVec3

#Calculate r1Vec, r2Vec, r3Vec
def calcrVec(rhoVec1, rhoVec2, rhoVec3, Rvec1, Rvec2, Rvec3):
    rVec1 = rhoVec1-Rvec1
    rVec2 = rhoVec2-Rvec2
    rVec3 = rhoVec3-Rvec3
    return rVec1, rVec2, rVec3

#Calculate r2VecDot
def calcr2VecDot(rVec1, rVec3, f1, f3, g1, g3):
    r2VecDot = (f3*rVec1-f1*rVec3)/(g1*f3-f1*g3)
    return r2VecDot

#Calculate delta E
def deltaENewtonMethod(a, T, r2, r2Dot):
    r2Mag = np.linalg.norm(r2)
    r2DotMag = np.linalg.norm(r2Dot)
    n = sqrt(mu/a**3)
    x = n*T
    xPrev = 1 + x
    while abs(xPrev - x) > 1e-004:
        xPrev = x
        f = x-(1-r2Mag/a)*sin(x)+((r2Mag)*r2DotMag/(n*a**2))*(1-cos(x))-(n*T)
        fPrime = 1-(1-(r2Mag/a))*cos(x)+((r2Mag)*r2DotMag/(n*a**2))*sin(x)
        x = x - f/fPrime
    return x

#Baby OD Function
def calcBabyOD(rPos, rVel): 

    G=6.674e-11 #Gravitational Constant

    #calculate a
    rMag=np.linalg.norm(rPos)
    a=(2/rMag-rVel.dot(rVel))**-1

    #calculate e
    rPosVelCross=np.cross(rPos,rVel) #cross product of r position and r velocity
    magrPosVelCross=np.linalg.norm(rPosVelCross)
    e=sqrt(1-(magrPosVelCross**2)/a)

    #calculate I
    I=acos(rPosVelCross[2]/magrPosVelCross)

    #calculate Omega
        #calculate mean anomaly
    v=degrees(atan2(((a*(1-e**2))/(e*magrPosVelCross))*(rPos.dot(rVel)/rMag), (1/e)*(((a*(1-e**2))/rMag)-1)))%360
    Om=atan2((I*rPosVelCross[0])/(magrPosVelCross*sin(I)), -(I*rPosVelCross[1])/(magrPosVelCross*sin(I)))

    #calculate omega
        #calculate f+w
    sinFPlusw=rPos[2]/(rMag*sin(I))
    cosFPlusw=(1/cos(Om))*(rPos[0]/rMag+cos(I)*sinFPlusw*sin(Om))
    fPlusw=atan2(sinFPlusw, cosFPlusw)
        #calculate f
    f=atan2(((a*(1-e**2))/magrPosVelCross)*(rPos.dot(rVel)/(e*rMag)), (1/e)*((a*(1-e**2))/rMag-1))
    w=fPlusw-f

    #calculate Mnot
    t=2457946.5

    #calculate E
    E=acos((1/e)*(1-(rMag/a)))
    M=E-e*sin(E)
    if v>180:
        M=2*pi-M
    return a, e, I, Om, w, M
