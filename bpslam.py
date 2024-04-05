# SLAM Gaussian Belief Propagation
# Andrew Davison, Imperial College London, 2019

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pygame

import random
from pygame.locals import *
pygame.init()
        
 
# set the width and height of the screen (pixels)
WIDTH = 1000
HEIGHT = 1000

size = [WIDTH, HEIGHT]
black = (20,20,40)
green = (0,255,0)
dullgreen = (0,180,0)
lightblue = (0,120,255)
darkblue = (0,40,160)
red = (255,100,0)
yellow = (255,255,0)
white = (255,255,255)
blue = (0,0,255)
grey = (70,70,70)
lightgrey = (150,150,150)



# Screen centre will correspond to (x, y) = (0, 0)
u0 = 100
v0 = HEIGHT - 100

POINTSXRANGE = 10.0
POINTSYRANGE = 10.0
NUMBEROFPOINTS = 30


kx = (WIDTH - 200) / float(POINTSXRANGE)

# For printing text
myfont = pygame.font.SysFont("Jokerman", 30)

# Standard deviations
MEASUREMENTsigma = 0.05
POINTPOSEsigma = 100.0
ODOMETRYMEASUREMENTsigma = 0.1
ROBOTINITIALPOSEsigma = 0.0001
# Inverse of this is outlier fraction in measurements
OUTLIERRATIO = 100000000000

ROBOTSTARTX = 0.0
ROBOTSTARTY = 0.0

# Robots will measure every point within this distance
MAXMEASDISTANCE = 2.0

# For robust cost function
NSIGMA = 10000000.0


# For batch solution
bigSigma = -1
bigmu = -1
batchvalidflag = 0

pygame.init()



# Initialise Pygame display screen
screen = pygame.display.set_mode(size)
# This makes the normal mouse pointer invisible in graphics window


# For Huber: this is factor by which we reduce energy of factor if Mahalanobis
# distance Dmtest is above Nsigma
def kRHuber(Dmtest, Nsigma):
        if (abs(Dmtest) > Nsigma):
                return 2.0 * Nsigma / Dmtest - (Nsigma**2)/Dmtest**2
        else:
                return 1.0

# For robust energy where energy is constant after Nsigma
def kRCutoff(Dmtest, Nsigma):
        if (abs(Dmtest) > Nsigma):
                return (Nsigma**2)/Dmtest**2
        else:
                return 1.0


        

class GroundTruthPoint:
        def __init__(self, index, x, y):
                self.index = index
                self.x = np.matrix([[x],[y]])
                print ("New ground truth point", index, "with position", x, y)



class GroundTruthMeasurement:
        def __init__(self, fromgroundtruthpoint, togroundtruthpoint, sigma):
                self.fromgroundtruthpoint = fromgroundtruthpoint
                self.togroundtruthpoint = togroundtruthpoint
                self.ztrue = togroundtruthpoint.x - fromgroundtruthpoint.x
                self.znoisy = self.ztrue.copy()
                sigmause = sigma
                outlierflag = random.randint(0,OUTLIERRATIO)
                if (outlierflag == 0):
                        sigmause = sigma * 100.0
                self.znoisy[0,0] += random.gauss(0.0, sigmause)
                self.znoisy[1,0] += random.gauss(0.0, sigmause)
                # Evaluate Mahalanobis distance for this ground truth noisy measurement
                Sigma = np.matrix([[sigma * sigma, 0.0], [0.0, sigma * sigma]])
                err = self.znoisy - self.ztrue
                self.Dm = math.sqrt(err.T * Sigma.I * err)
                print ("New factor from point", fromgroundtruthpoint.index, "to point", togroundtruthpoint.index)
                print ("Ground truth", self.ztrue, "Noisy", self.znoisy)






class VariableNode:
        def __init__(self, index):
                self.index = index
                # mu and Sigma are stored for visualisation purposes, etc., but are not the master data here
                self.mu = np.matrix([[0.0], [0.0]])
                self.Sigma = np.matrix([[10.0, 0.0], [0.0, 10.0]])

                # List of links to edges
                self.edges = []
                self.updatedflag = 0

        def sendAllMessages(self):
                #print("Sending all messages: could be implemented much more efficiently.")
                for (i, edge) in enumerate(self.edges):
                        self.sendMessage(i)

        def updatemuSigma(self):
                # Just update the mu mean estimate from all current incoming messsages
                eta = np.matrix([[0.0], [0.0]])
                Lambdaprime = np.matrix([[0.0, 0.0], [0.0, 0.0]])

                # Multiply inward messages from other factors
                for i, edge in enumerate(self.edges):
                        # Multiply incoming messages by adding precisions
                        (etainward, Lambdaprimeinward) = (self.edges[i].factortovariablemessage.eta, self.edges[i].factortovariablemessage.Lambda)
                        eta += etainward
                        Lambdaprime += Lambdaprimeinward        
                        
                self.Sigma = Lambdaprime.I
                self.mu = self.Sigma * eta
                

                
        # localfactorID is factor we send message out to
        def sendMessage(self, outedgeID):
                #print ("Sending message from variable", self.index, "to edge with local ID", outedgeID, "global ID", self.edges[outedgeID].edgeID)
                eta = np.matrix([[0.0], [0.0]])
                Lambdaprime = np.matrix([[0.0, 0.0], [0.0, 0.0]])

                # Multiply inward messages from other factors
                for i, edge in enumerate(self.edges):
                        if (i != outedgeID):
                                # Multiply incoming messages by adding precisions
                                #print ("Multiplying inward message from edge with local ID", i, "global ID", edge.edgeID)
                                (etainward, Lambdaprimeinward) = (self.edges[i].factortovariablemessage.eta, self.edges[i].factortovariablemessage.Lambda)
                                eta += etainward
                                Lambdaprime += Lambdaprimeinward        
                                #print ("eta, Lambdaprime", eta, Lambdaprime)
                self.edges[outedgeID].variabletofactormessage.eta = eta.copy()
                self.edges[outedgeID].variabletofactormessage.Lambda = Lambdaprime.copy()
                #print (                self.edges[outedgeID].VariableToFactorMessage)
                #print ("Message", self.edges[outedgeID].VariableToFactorMessage[0],self.edges[outedgeID].VariableToFactorMessage[1])
                
                # Update local state estimate from all messages including outward one
                eta += self.edges[outedgeID].factortovariablemessage.eta
                Lambdaprime += self.edges[outedgeID].factortovariablemessage.Lambda
                if (Lambdaprime[0,0] != 0.0):
                        self.Sigma = Lambdaprime.I
                self.mu = self.Sigma * eta
                self.edges[outedgeID].variabletofactormessage.mu = self.mu.copy()


class VariableToFactorMessage:
        def __init__(self, dimension):
                self.eta = np.zeros((dimension, 1))
                self.Lambda = np.zeros((dimension, dimension))
                self.mu = np.zeros((dimension, 1)) 

class FactorToVariableMessage:
        def __init__(self, dimension):
                self.eta = np.zeros((dimension, 1))
                self.Lambda = np.zeros((dimension, dimension))


# Every edge between a variable node and a factor node is via an edge
# So the variable and factor only need to know about the edge, not about
# each other's internals
# An edge stores the recent message in each direction; dimension is the state space size
class Edge:
        def __init__(self, dimension):
                # VariableToFactorMessage also passes the current state value (3rd arg)
                self.variabletofactormessage = VariableToFactorMessage(dimension)
                self.factortovariablemessage = FactorToVariableMessage(dimension)
                self.variable = -1
                self.factor = -1

# What we need to specify for a factor:
# Number of variables inputs and size of each: implicit in edges
# Measurement (vector with size)
# h (function of x)
# J (function of x)
# Lambda matrix
# Write generalised code for marginalisation
class FactorNode:
        def __init__(self, factorID, edges, z):
                self.factorID = factorID
                self.edges = edges
                self.z = z
                self.etastored = -1
                self.Lambdaprimestored = -1
                self.recalculateetaLambdaprime()
                print ("New FactorNode factorID = ", factorID)

        # Get current overall stacked state vector estimate from input edges
        def x0(self):
                for (i, edge) in enumerate(self.edges):
                        if (i==0):
                                x0local = edge.variabletofactormessage.mu
                        else:
                                x0local = np.concatenate((x0local, edge.variabletofactormessage.mu), axis=0)
                return x0local

        
        def sendAllMessages(self):
                #print("Sending all messages: could be implemented much more efficiently.")
                # Recalculate factor every time; not really needed in purely linear case!
                self.recalculateetaLambdaprime()

                for (i,edge) in enumerate(self.edges):
                        self.sendMessage(i)

        # Relinearise factor if needed (with linear factor only need to call once)
        def recalculateetaLambdaprime(self):
                # First form eta and Lambdaprime for linearised factor
                # Recalculate h and J in principle so we can relinearise
                hlocal = self.h()
                Jlocal = self.J()
                self.Lambdaprimestored = Jlocal.T * self.Lambda * Jlocal
                self.etastored = Jlocal.T * self.Lambda * (Jlocal * self.x0() + self.z - hlocal)
                # Calculate Mahalanobis distance Dm
                err = self.z - hlocal
                self.Dm = math.sqrt(err.T * self.Lambda * err)
                
                        
        def sendMessage(self, outedgeID):
                #print ("Sending message from factor", self.factorID, "to edge with local ID", outedgeID)
                #For Huber
                kRlocal = kRHuber(self.Dm, NSIGMA)
                
                # Copy factor precision vector and matrix from stored; could recalculate every time
                # here if needed.
                eta = self.etastored.copy() * kRlocal
                Lambdaprime = self.Lambdaprimestored.copy() * kRlocal
                
                # Next condition factors on messages from all edges apart from outward one
                offset = 0
                outoffset = 0
                outsize = 0
                for (i, edge) in enumerate(self.edges):
                        edgeeta = edge.variabletofactormessage.eta
                        edgeLambda = edge.variabletofactormessage.Lambda
                        edgesize = edgeeta.shape[0]
                        if (i != outedgeID):
                                # For edges which are not the outward one, condition
                                eta[offset:offset+edgesize,:] += edgeeta
                                Lambdaprime[offset:offset+edgesize,offset:offset+edgesize] += edgeLambda
                        else:
                                # Remember where in the matrix the outward one is
                                outoffset = offset
                                outsize = edgesize
                        offset += edgesize

                # Now restructure eta and Lambdaprime so the outward variable of interest is at the top
                etatoptemp = eta[0:outsize,:].copy()
                eta[0:outsize,:] = eta[outoffset:outoffset+outsize,:]
                eta[outoffset:outoffset+outsize,:] = etatoptemp
                Lambdaprimetoptemp = Lambdaprime[0:outsize,:].copy()
                Lambdaprime[0:outsize,:] = Lambdaprime[outoffset:outoffset+outsize,:]
                Lambdaprime[outoffset:outoffset+outsize,:] = Lambdaprimetoptemp
                Lambdaprimelefttemp = Lambdaprime[:,0:outsize].copy()
                Lambdaprime[:,0:outsize] = Lambdaprime[:,outoffset:outoffset+outsize]
                Lambdaprime[:,outoffset:outoffset+outsize] = Lambdaprimelefttemp


                # To marginalise, first set up subblocks as in Eustice
                ea = eta[0:outsize,:]
                eb = eta[outsize:,:]
                aa = Lambdaprime[0:outsize,0:outsize]
                ab = Lambdaprime[0:outsize,outsize:]
                ba = Lambdaprime[outsize:,0:outsize]
                bb = Lambdaprime[outsize:,outsize:]

                bbinv = bb.I
                emarg = ea - ab * bbinv * eb
                Lmarg = aa - ab * bbinv * ba

                for row in range(outsize):
                        self.edges[outedgeID].factortovariablemessage.eta[row,0] = emarg[row,0]
                        for col in range(outsize):
                                self.edges[outedgeID].factortovariablemessage.Lambda[row,col] = Lmarg[row,col]


# Factor for relative measurement between points
class TwoDMeasFactorNode(FactorNode):
        def __init__(self, factorID, edges, znoisy, sigma):
                # Calls constructor of parent class
                self.Lambda = np.matrix(  [[1.0 / (sigma * sigma), 0.0], [0.0, 1.0 / (sigma * sigma)]]  )
                super().__init__(factorID, edges, znoisy)
                
        def h(self):
                #print ("Function to generate h.")
                xfrom = self.edges[0].variabletofactormessage.mu
                xto = self.edges[1].variabletofactormessage.mu
                h = xto - xfrom
                return h
                
        def J(self):
                #print ("Function to generate J.")
                # In general J could depend on variables but here it is constant because the measurement function is linear
                J = np.matrix( [[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]] )
                return J

        def setLambda(self, newMEASUREMENTsigma):
                self.Lambda = np.matrix(  [[1.0 / (newMEASUREMENTsigma * newMEASUREMENTsigma), 0.0], [0.0, 1.0 / (newMEASUREMENTsigma * newMEASUREMENTsigma)]]  )
                
# Simple factor for pose anchor measurement of a single point
class TwoDPoseFactorNode(FactorNode):
        def __init__(self, factorID, edges, poseobserved, Psigma):
                # Calls constructor of parent class
                self.Lambda = np.matrix(  [[1.0 / (Psigma * Psigma), 0.0], [0.0, 1.0 / (Psigma * Psigma)]]  )
                super().__init__(factorID, edges, poseobserved)
                
        def h(self):
                #print ("Function to generate h.")
                x = self.edges[0].variabletofactormessage.mu
                h = x
                return h
                
        def J(self):
                #print ("Function to generate J.")
                # In general J could depend on variables but here it is constant because the measurement function is linear
                J = np.matrix( [[1.0, 0.0], [0.0, 1.0]] )
                return J
                       
                

# Flag for display code
wasdflag = 1
rflag = 1
colourcodeflag = 0



                
def updateDisplay():
        screen.fill((0,0,0))

        # Draw factor colours based on Mahalonobis distance check
        for (i, groundtruthodometryfactor) in enumerate(groundtruthodometryfactors):
                xfrom = groundtruthodometryfactor.fromgroundtruthpoint.x[0,0]
                yfrom = groundtruthodometryfactor.fromgroundtruthpoint.x[1,0]
                xto = groundtruthodometryfactor.togroundtruthpoint.x[0,0]
                yto = groundtruthodometryfactor.togroundtruthpoint.x[1,0]
                xmid = (xfrom + xto)/2.0
                ymid = (yfrom + yto)/2.0
                if (odometryfactornodes[i].Dm <= NSIGMA and groundtruthodometryfactor.Dm <= NSIGMA):
                        colour = lightgrey
                else:
                        if(odometryfactornodes[i].Dm > NSIGMA and groundtruthodometryfactor.Dm > NSIGMA):
                                colour = white
                        elif (odometryfactornodes[i].Dm <= NSIGMA and groundtruthodometryfactor.Dm > NSIGMA):
                                colour = red
                        else:
                                colour = yellow
                        ssshow = myfont.render("%.1f %.1f" %(odometryfactornodes[i].Dm, groundtruthodometryfactor.Dm), True, colour)
                        screen.blit(ssshow, (u0 + kx * xmid, v0 - kx * ymid))
                pygame.draw.line(screen, colour, ( int(u0 + kx * xfrom), int(v0 - kx * yfrom) ), ( int(u0 + kx * xto), int(v0 - kx * yto)), 2)

        for (i, groundtruthmeasurementfactor) in enumerate(groundtruthmeasurementfactors):
                xfrom = groundtruthmeasurementfactor.fromgroundtruthpoint.x[0,0]
                yfrom = groundtruthmeasurementfactor.fromgroundtruthpoint.x[1,0]
                xto = groundtruthmeasurementfactor.togroundtruthpoint.x[0,0]
                yto = groundtruthmeasurementfactor.togroundtruthpoint.x[1,0]
                xmid = (xfrom + xto)/2.0
                ymid = (yfrom + yto)/2.0
                if (measurementfactornodes[i].Dm <= NSIGMA and groundtruthmeasurementfactor.Dm <= NSIGMA):
                        colour = lightgrey
                else:
                        if(measurementfactornodes[i].Dm > NSIGMA and groundtruthmeasurementfactor.Dm > NSIGMA):
                                colour = white
                        elif (measurementfactornodes[i].Dm <= NSIGMA and groundtruthmeasurementfactor.Dm > NSIGMA):
                                colour = red
                        else:
                                colour = yellow
                        ssshow = myfont.render("%.1f %.1f" %(measurementfactornodes[i].Dm, groundtruthmeasurementfactor.Dm), True, colour)
                        screen.blit(ssshow, (u0 + kx * xmid, v0 - kx * ymid))
                pygame.draw.line(screen, colour, ( int(u0 + kx * xfrom), int(v0 - kx * yfrom) ), ( int(u0 + kx * xto), int(v0 - kx * yto)), 2)



        for groundtruthpoint in groundtruthpoints:
                pygame.draw.circle(screen, white, ( int(u0 + kx * groundtruthpoint.x[0,0]), int( v0 - kx * groundtruthpoint.x[1,0] ) ), 6, 0)

        for pointvariablenode in pointvariablenodes:
                if (pointvariablenode.updatedflag == 1):
                        drawcolour = green
                else:
                        drawcolour = lightblue
                pygame.draw.circle(screen, drawcolour, (int(u0 + kx * pointvariablenode.mu[0,0]), int(v0 - kx * pointvariablenode.mu[1,0])), 4, 0)
                sigma = math.sqrt(pointvariablenode.Sigma[0,0])
                if(sigma * kx > 1.0):
                        pygame.draw.circle(screen, lightblue, (int(u0 + kx * pointvariablenode.mu[0,0]), int(v0 - kx * pointvariablenode.mu[1,0])), int(sigma * kx), 1)                




                
        for groundtruthrobot in groundtruthrobots:
                pygame.draw.circle(screen, yellow, ( int(u0 + kx * groundtruthrobot.x[0,0]), int( v0 - kx * groundtruthrobot.x[1,0] ) ), 6, 0)

        for robotvariablenode in robotvariablenodes:
                pygame.draw.circle(screen, red, (int(u0 + kx * robotvariablenode.mu[0,0]), int(v0 - kx * robotvariablenode.mu[1,0])), 4, 0)
                sigma = math.sqrt(robotvariablenode.Sigma[0,0])
                if(sigma * kx > 1.0):
                        pygame.draw.circle(screen, red, (int(u0 + kx * robotvariablenode.mu[0,0]), int(v0 - kx * robotvariablenode.mu[1,0])), int(sigma * kx), 1)                


        if (batchvalidflag == 1):
                # Draw optimal batch solution
                robotstatesize = 2 * len(robotvariablenodes)
                for (i,robotvariablenode) in enumerate(robotvariablenodes):
                        (bx, by) = (bigmu[2*i,0],bigmu[2*i+1,0])
                        sigma = math.sqrt(bigSigma[2*i,2*i])
                        pygame.draw.circle(screen, dullgreen, (int(u0 + kx * bx), int(v0 - kx * by)), 4, 0)
                        if(sigma * kx > 1.0):
                                pygame.draw.circle(screen, dullgreen, (int(u0 + kx * bx), int(v0 - kx * by)), int(sigma * kx), 1)
                for (i,pointvariablenode) in enumerate(pointvariablenodes):
                        (bx, by) = (bigmu[robotstatesize+2*i,0],bigmu[robotstatesize+2*i+1,0])
                        sigma = math.sqrt(bigSigma[robotstatesize+2*i,robotstatesize+2*i])
                        pygame.draw.circle(screen, green, (int(u0 + kx * bx), int(v0 - kx * by)), 4, 0)                
                        if(sigma * kx > 1.0):
                                pygame.draw.circle(screen, green, (int(u0 + kx * bx), int(v0 - kx * by)), int(sigma * kx), 1)                

                        
        ssshow = myfont.render("w,a,s,d: move robot", True, white)
        screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 20))
        if (rflag):
                ssshow = myfont.render("r: turn on error measurements and robust factors", True, white)
                screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 60))
        else:
                ssshow = myfont.render("unrecognised bad measurements", True, red)
                screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 80))
                ssshow = myfont.render("rejected good measurements", True, yellow)
                screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 100))                
                ssshow = myfont.render("rejected bad measurements", True, white)
                screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 120))                
                ssshow = myfont.render("accepted good measurements", True, grey)
                screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 140))                
        if (not batchvalidflag):
                ssshow = myfont.render("b: calculate non-robust batch solution", True, green)
                screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 40))

        pygame.display.flip()







# Implement optimal batch solution
def calculateBatchSolution():

        # Form big matrices with robots first, points second
        robotstatesize = 2 * len(robotvariablenodes)
        totalstatesize = 2 * (len(robotvariablenodes) + NUMBEROFPOINTS)

        bigeta = np.zeros((totalstatesize, 1))
        bigLambda = np.zeros((totalstatesize, totalstatesize))

        # Probably only one of these
        for (i,robotposefactornode) in enumerate(robotposefactornodes):
                print ("robotposefactornode", i)
                bigeta[2*i:2*i+2,:] = robotposefactornode.etastored.copy()
                bigLambda[2*i:2*i+2,2*i:2*i+2] = robotposefactornode.Lambdaprimestored.copy()

                print (bigeta)
                print (bigLambda)

        for (i,pointposefactornode) in enumerate(pointposefactornodes):
                print ("pointposefactornode", i)
                bigeta[robotstatesize+2*i:robotstatesize+2*i+2,:] = pointposefactornode.etastored.copy()
                bigLambda[robotstatesize+2*i:robotstatesize+2*i+2,robotstatesize+2*i:robotstatesize+2*i+2] = pointposefactornode.Lambdaprimestored.copy()

                print (bigeta)
                print (bigLambda)


        for (i,odometryfactornode) in enumerate(odometryfactornodes):
                fromindex = odometryfactornode.edges[0].variable.index
                toindex = odometryfactornode.edges[1].variable.index
                print ("odometryfactornode fromindex", fromindex, "toindex", toindex)
                etablock = odometryfactornode.etastored.copy()
                Lambdablock = odometryfactornode.Lambdaprimestored.copy()
                print (etablock)
                print (Lambdablock)
                bigeta[2*fromindex:2*fromindex+2,:] += etablock[0:2,:]
                bigeta[2*toindex:2*toindex+2,:] += etablock[2:4,:]
                bigLambda[2*fromindex:2*fromindex+2,2*fromindex:2*fromindex+2] += Lambdablock[0:2,0:2]
                bigLambda[2*fromindex:2*fromindex+2,2*toindex:2*toindex+2] += Lambdablock[0:2,2:4]
                bigLambda[2*toindex:2*toindex+2,2*fromindex:2*fromindex+2] += Lambdablock[2:4,0:2]
                bigLambda[2*toindex:2*toindex+2,2*toindex:2*toindex+2] += Lambdablock[2:4,2:4]


                print (bigeta)
                print (bigLambda)

        for (i,measurementfactornode) in enumerate(measurementfactornodes):
                robotfromindex = measurementfactornode.edges[0].variable.index
                pointtoindex = measurementfactornode.edges[1].variable.index
                print ("measurementfactornode robotfromindex", robotfromindex, "pointtoindex", pointtoindex)
                etablock = measurementfactornode.etastored.copy()
                Lambdablock = measurementfactornode.Lambdaprimestored.copy()
                print (etablock)
                print (Lambdablock)
                bigeta[2*robotfromindex:2*robotfromindex+2,:] += etablock[0:2,:]
                bigeta[robotstatesize+2*pointtoindex:robotstatesize+2*pointtoindex+2,:] += etablock[2:4,:]
                bigLambda[2*robotfromindex:2*robotfromindex+2,2*robotfromindex:2*robotfromindex+2] += Lambdablock[0:2,0:2]
                bigLambda[2*robotfromindex:2*robotfromindex+2,robotstatesize+2*pointtoindex:robotstatesize+2*pointtoindex+2] += Lambdablock[0:2,2:4]
                bigLambda[robotstatesize+2*pointtoindex:robotstatesize+2*pointtoindex+2,2*robotfromindex:2*robotfromindex+2] += Lambdablock[2:4,0:2]
                bigLambda[robotstatesize+2*pointtoindex:robotstatesize+2*pointtoindex+2,robotstatesize+2*pointtoindex:robotstatesize+2*pointtoindex+2] += Lambdablock[2:4,2:4]

                print ("bigeta")
                print (bigeta)
                print ("biglambda")
                print (bigLambda)


        global bigSigma
        global bigmu
        bigSigma = np.matrix(bigLambda).I.copy()
        bigmu = bigSigma * bigeta

        print ("bigmu")
        print (bigmu)
        print ("bigSigma")
        print (bigSigma)
















        
groundtruthpoints = []
pointvariablenodes = []
pointposefactornodes = []


groundtruthrobots = []
robotvariablenodes = []
robotposefactornodes = []

groundtruthodometryfactors = []
odometryfactornodes = []

groundtruthmeasurementfactors = []
measurementfactornodes = []




# For SLAM operation, set up a number of points that are available to map, with weak absolute pose factors


print ("Setting up point nodes.")


# Make sure there is at least one point close to the robot at the start
for i in range(NUMBEROFPOINTS):
        if (i == 0):
                rrange = 1.5
        else:
                rrange = POINTSXRANGE
        x = random.uniform (0.0, rrange)
        y = random.uniform (0.0, rrange)
        groundtruthpoints.append(GroundTruthPoint(i, x, y))
        pointvariablenodes.append(VariableNode(i))

        poseedges = []
        newposeedge = Edge(2)
        poseedges.append(newposeedge)
        newpointposefactornode = TwoDPoseFactorNode(i, poseedges, groundtruthpoints[i].x, POINTPOSEsigma)
        pointposefactornodes.append(newpointposefactornode)
        pointvariablenodes[i].edges.append(newposeedge)
        newposeedge.variable = pointvariablenodes[i]
        newposeedge.factor = newpointposefactornode


print ("Setting up robot nodes.")


# Set up first robot node
groundtruthrobots.append(GroundTruthPoint(0, ROBOTSTARTX, ROBOTSTARTY))
robotvariablenodes.append(VariableNode(0))
poseedges = []
newposeedge = Edge(2)
poseedges.append(newposeedge)
newrobotposefactornode = TwoDPoseFactorNode(0, poseedges, groundtruthrobots[0].x, ROBOTINITIALPOSEsigma)
robotposefactornodes.append(newrobotposefactornode)
robotvariablenodes[0].edges.append(newposeedge)
newposeedge.variable = robotvariablenodes[0]
newposeedge.factor = newrobotposefactornode
                         

for pointposefactornode in pointposefactornodes:
        pointposefactornode.sendAllMessages()

for pointvariablenode in pointvariablenodes:
        pointvariablenode.updatemuSigma()

robotposefactornodes[0].sendAllMessages()
robotvariablenodes[0].updatemuSigma()



# See if this new robot is close to any points and simulate measurements
for groundtruthpoint in groundtruthpoints:
        newgroundtruthmeasurementfactor = GroundTruthMeasurement(groundtruthrobots[0], groundtruthpoint, MEASUREMENTsigma)
        if (np.linalg.norm(newgroundtruthmeasurementfactor.ztrue) < MAXMEASDISTANCE):
                groundtruthmeasurementfactors.append(newgroundtruthmeasurementfactor)

                # Set up factor node
                newedges = []
                fromedge = Edge(2)
                toedge = Edge(2)
                newedges.append(fromedge)
                newedges.append(toedge)
                newtwodmeasfactornode = TwoDMeasFactorNode(len(measurementfactornodes), newedges, newgroundtruthmeasurementfactor.znoisy, MEASUREMENTsigma)
                measurementfactornodes.append(newtwodmeasfactornode)
                robotvariablenodes[0].edges.append(fromedge)
                pointvariablenodes[groundtruthpoint.index].edges.append(toedge)
                fromedge.variable = robotvariablenodes[0]
                fromedge.factor = newtwodmeasfactornode
                toedge.variable = pointvariablenodes[groundtruthpoint.index]
                toedge.factor = newtwodmeasfactornode



speed = 0.6
                
delta_x = speed
delta_y = 0.0



count = 0

while (1):
        Eventlist = pygame.event.get()


        movementflag = 0

        for event in Eventlist:
                if (event.type == KEYDOWN):
                        if (event.key == K_b):
                                calculateBatchSolution()
                                batchvalidflag = 1
                        if (event.key == K_w):
                                movementflag = 1
                                bigSigma = -1
                                delta_x = 0.0
                                delta_y = speed
                        if (event.key == K_a):
                                movementflag = 1
                                bigmu = -1
                                bigSigma = -1
                                delta_x = -speed
                                delta_y = 0.0
                        if (event.key == K_s):
                                movementflag = 1
                                bigmu = -1
                                bigSigma = -1
                                delta_x = 0.0
                                delta_y = -speed
                        if (event.key == K_d):
                                movementflag = 1
                                bigmu = -1
                                bigSigma = -1
                                delta_x = speed
                                delta_y = 0.0
                        if (event.key == K_r):
                                rflag = 0
                                NSIGMA = 4.0
                                OUTLIERRATIO = 50
                        if (event.key == K_i):
                                myfile = "2drobust" + str(count) + ".png"
                                pygame.image.save(screen, myfile)
                                print ("Saving image to", myfile)
                                
        if (movementflag):
                batchvalidflag = 0
                
                # Initialise new robot node 
                lastgroundtruthrobot = groundtruthrobots[-1]
                lastrobotvariablenode = robotvariablenodes[-1]
                newgroundtruthrobot = GroundTruthPoint(lastgroundtruthrobot.index + 1, lastgroundtruthrobot.x[0,0] + delta_x, lastgroundtruthrobot.x[1,0] + delta_y)
                groundtruthrobots.append(newgroundtruthrobot)
                newrobotvariablenode = VariableNode(lastrobotvariablenode.index + 1)
                robotvariablenodes.append(newrobotvariablenode)

                
                # Set up odometry factor linking it to last robot node
                newgroundtruthodometryfactor = GroundTruthMeasurement(lastgroundtruthrobot, newgroundtruthrobot,ODOMETRYMEASUREMENTsigma)
                groundtruthodometryfactors.append(newgroundtruthodometryfactor)
                newedges = []
                fromedge = Edge(2)
                toedge = Edge(2)
                newedges.append(fromedge)
                newedges.append(toedge)
                newtwododometryfactornode = TwoDMeasFactorNode(lastrobotvariablenode.index, newedges, newgroundtruthodometryfactor.znoisy,ODOMETRYMEASUREMENTsigma)
                odometryfactornodes.append(newtwododometryfactornode)
                robotvariablenodes[lastrobotvariablenode.index].edges.append(fromedge)
                robotvariablenodes[lastrobotvariablenode.index + 1].edges.append(toedge)
                fromedge.variable = robotvariablenodes[lastrobotvariablenode.index]
                fromedge.factor = newtwododometryfactornode
                toedge.variable = robotvariablenodes[lastrobotvariablenode.index + 1]
                toedge.factor = newtwododometryfactornode

                # Pass messages into new robot node
                lastrobotvariablenode.sendAllMessages()
                newtwododometryfactornode.sendAllMessages()
                newrobotvariablenode.updatemuSigma()



                # See if this new robot is close to any points and simulate measurements
                for groundtruthpoint in groundtruthpoints:
                        newgroundtruthmeasurementfactor = GroundTruthMeasurement(newgroundtruthrobot, groundtruthpoint, MEASUREMENTsigma)
                        if (np.linalg.norm(newgroundtruthmeasurementfactor.ztrue) < MAXMEASDISTANCE):
                                groundtruthmeasurementfactors.append(newgroundtruthmeasurementfactor)

                                # Set up factor node
                                newedges = []
                                fromedge = Edge(2)
                                toedge = Edge(2)
                                newedges.append(fromedge)
                                newedges.append(toedge)
                                newtwodmeasfactornode = TwoDMeasFactorNode(len(measurementfactornodes), newedges, newgroundtruthmeasurementfactor.znoisy, MEASUREMENTsigma)
                                measurementfactornodes.append(newtwodmeasfactornode)
                                newrobotvariablenode.edges.append(fromedge)
                                pointvariablenodes[groundtruthpoint.index].edges.append(toedge)
                                fromedge.variable = newrobotvariablenode
                                fromedge.factor = newtwodmeasfactornode
                                toedge.variable = pointvariablenodes[groundtruthpoint.index]
                                toedge.factor = newtwodmeasfactornode

        for robotvariablenode in robotvariablenodes:
                robotvariablenode.sendAllMessages()

        for pointvariablenode in pointvariablenodes:
                pointvariablenode.sendAllMessages()

        for odometryfactornode in odometryfactornodes:
                odometryfactornode.sendAllMessages()

        for measurementfactornode in measurementfactornodes:
                measurementfactornode.sendAllMessages()
                
        for robotvariablenode in robotvariablenodes:
                robotvariablenode.updatemuSigma()
        for pointvariablenode in pointvariablenodes:
                pointvariablenode.updatemuSigma()


        count += 1

        updateDisplay()

        #time.sleep(1.0)
