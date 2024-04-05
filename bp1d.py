# 1D Gaussian Belief Propagation
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
WIDTH = 1500
HEIGHT = 1000

size = [WIDTH, HEIGHT]
black = (20,20,40)
lightblue = (0,120,255)
darkblue = (0,40,160)
red = (255,100,0)
white = (255,255,255)
blue = (0,0,255)
grey = (70,70,70)

# Screen centre will correspond to (x, y) = (0, 0)
u0 = 100
v0 = HEIGHT - 100

NUMBEROFVARIABLES = 41

k = (WIDTH - 200) / float(NUMBEROFVARIABLES - 1) # pixels per metre for graphics

# For printing text
myfont = pygame.font.SysFont("Jokerman", 40)


# Standard deviations
MEASUREMENTsigma = 0.6
SMOOTHNESSsigma = 0.6



pygame.init()



# Initialise Pygame display screen
screen = pygame.display.set_mode(size)
# This makes the normal mouse pointer invisible in graphics window
#pygame.mouse.set_visible(0)

        
        
def MeasurementFactor(x1, x2, z, x0, Lambda):
        lambda0 = (x0 - x1) / (x2 - x1)
        #print ("lambda0", lambda0) 
        J = np.matrix([[1-lambda0, lambda0]])
        #print ("J", J)
        # Precision in y1, y2 space
        Lambdaprime = Lambda * J.T * J
        eta = Lambda * z * J.T
        return (eta, Lambdaprime)
        

def SmoothnessFactor(Lambda):
        J = np.matrix([[-1, 1]])
        # Precision in y1, y2 space
        Lambdaprime = Lambda * J.T * J
        eta = Lambda * 0.0 * J.T
        return (eta, Lambdaprime)
        

# Class defines what a factor does; multiple factors over the same
# variables can be combined into it
class TwoVariableFactor:
        def __init__ (self):
                #print ("Initialising empty TwoVariableFactor.")

                self.eta = np.matrix([[0.0],[0.0]])
                self.Lambdaprime = np.matrix([[0.0, 0.0],[0.0,0.0]])
        
        
# Every edge between a variable and a node is via a message holder
# So the variable and node only need to know about the message holder, not about
# each other's internals
class Edge:
        def __init__(self, edgeID):
                self.edgeID = edgeID
                self.VariableToFactorMessage = (np.matrix([[0.0]]), np.matrix([[0.0]]) )
                self.FactorToVariableMessage = (np.matrix([[0.0]]), np.matrix([[0.0]]) )
        
        
        
class VariableNode:
        def __init__(self, variableID, x, y):
                self.variableID = variableID
                self.x = x
                self.y = y
                self.mu = np.matrix([[float(y)]])
                self.Sigma = np.matrix([[10.0]])

                # List of links to edges
                self.edges = []

        def draw(self):
                pygame.draw.circle(screen, white, (int(u0 + k * self.x), int(v0 - k * self.y)), 4, 0)
                sigma = math.sqrt(self.Sigma[0,0])
                pygame.draw.line(screen, white, (int(u0 + k * self.x), int(v0 - k * (self.y + sigma))), (int(u0 + k * self.x), int(v0 - k * (self.y - sigma))), 2)

                
        # localfactorID is factor we send message out to
        def sendMessage(self, outedgeID):
                #print ("Sending message from variable", self.variableID, "to edge with local ID", outedgeID, "global ID", self.edges[outedgeID].edgeID)
                eta = np.matrix([[0.0]])
                Lambdaprime = np.matrix([[0.0]])

                # Multiply inward messages from other factors
                for i, edge in enumerate(self.edges):
                        if (i != outedgeID):
                                # Multiply incoming messages by adding precisions
                                #print ("Multiplying inward message from edge with local ID", i, "global ID", edge.edgeID)
                                (etainward, Lambdaprimeinward) = self.edges[i].FactorToVariableMessage 
                                eta += etainward
                                Lambdaprime += Lambdaprimeinward        
                                #print ("eta, Lambdaprime", eta, Lambdaprime)
                self.edges[outedgeID].VariableToFactorMessage[0][0,0] = eta[0,0]
                self.edges[outedgeID].VariableToFactorMessage[1][0,0] = Lambdaprime[0,0]
                #print (         self.edges[outedgeID].VariableToFactorMessage)
                #print ("Message", self.edges[outedgeID].VariableToFactorMessage[0],self.edges[outedgeID].VariableToFactorMessage[1])
                
                # Update local state estimate from all messages including outward one
                eta += self.edges[outedgeID].FactorToVariableMessage[0]
                Lambdaprime += self.edges[outedgeID].FactorToVariableMessage[1]
                if (Lambdaprime[0,0] != 0.0):
                        self.Sigma = Lambdaprime.I
                self.mu = self.Sigma * eta
                self.y = self.mu[0,0]
        
class FactorNode:
        def __init__(self, factorID, z):
                self.factorID = factorID
                self.z = z
                self.edges = []
                self.twovariablefactor = TwoVariableFactor()
        
        #def draw(self):
                #if (len(self.variables) == 2):
                        #x0 = self.variables[0].x
                        #y0 = self.variables[0].y
                        #x1 = self.variables[1].x
                        #y1 = self.variables[1].y
                        #pygame.draw.line(screen, red, (u0 + k * x0, v0 - k * y0), (u0 + k * x1, v0 - k * y1), 2)
                        #midx = (x0 + x1)/2.0
                        #midy = (y0 + y1)/2.0
                        #pygame.draw.rect(screen, red, ((u0 + k * (midx - 0.1), v0 - k * (midy + 0.1)), (0.2 * k, 0.2 * k)), 0)
        
        def sendMessage(self, outedgeID):
                #print ("Sending message from factor", self.factorID, "to edge with local ID", outedgeID, "global ID", self.edges[outedgeID].edgeID)
                etafactor = self.twovariablefactor.eta.copy()
                Lambdaprimefactor = self.twovariablefactor.Lambdaprime.copy()
                #print ("Standard etafactor, Lambaprimefactor", etafactor, Lambdaprimefactor)
                
                # Multiply by distributions from inward variables
                for i, edge in enumerate(self.edges):
                        if (i != outedgeID):
                                #print ("Multiplying factor by inward message from edge with local ID", i, "global ID", edge.edgeID)
                                etainward = self.edges[i].VariableToFactorMessage[0].copy()
                                Lambdainward = self.edges[i].VariableToFactorMessage[1].copy()
                                #print ("etainward, Lambdainward", etainward, Lambdainward)
                                etafactor[i,0] += etainward[0,0]
                                Lambdaprimefactor[i,i] += Lambdainward[0,0]
                                #print ("etafactor, Lambaprimefactor", etafactor, Lambdaprimefactor)
                # Marginalise for correct outward variable


                # First restructure eta and Lambdaprime until the outward variable of interest is at the top
                if (outedgeID != 0):
                        # Swap rows of etafactor
                        etafactor = etafactor[[outedgeID,0],:]

                        # I think this only works for two variable factors

                        # Swap rows of Lambdaprimefactor
                        Lambdaprimefactor = Lambdaprimefactor[[outedgeID,0],:]
                        # Swap cols of Lambdaprimefactor
                        Lambdaprimefactor = Lambdaprimefactor[:,[outedgeID,0]]

                #print ("Swapped:", etafactor, Lambdaprimefactor)

                # Marginalise
                ab = Lambdaprimefactor[0,1]
                ba = Lambdaprimefactor[1,0]
                bb = Lambdaprimefactor[1,1]
                aa = Lambdaprimefactor[0,0]
                ea = etafactor[0,0]
                eb = etafactor[1,0]
                e = ea - ab * eb / bb
                La = aa - ab * ba / bb
                self.edges[outedgeID].FactorToVariableMessage[0][0,0] = e
                self.edges[outedgeID].FactorToVariableMessage[1][0,0] = La
                #print ("Message", self.edges[outedgeID].FactorToVariableMessage[0],edges[outedgeID].FactorToVariableMessage[1])
                


def updateDisplay():
        # Do graphics
        screen.fill(black)

        for datapoint in measarray:
                x = datapoint[0,0]
                y = datapoint[1,0]
                pygame.draw.circle(screen, red, (int(u0 + k * x), int(v0 - k * y)), 6, 0)
                pygame.draw.line(screen, red, (int(u0 + k * x), int(v0 - k * (y + MEASUREMENTsigma))), (int(u0 + k * x), int(v0 - k * (y - MEASUREMENTsigma))), 2)

        for variablenode in variablenodes:
                variablenode.draw()



        if (keys_flag):
                ssshow = myfont.render("f: floodfill schedule", True, white)
                screen.blit(ssshow, (WIDTH - 200 - ssshow.get_width()/2, 20))
                ssshow = myfont.render("r: random schedule", True, white)
                screen.blit(ssshow, (WIDTH - 200 - ssshow.get_width()/2, 60))


        pygame.display.flip()



def randomSchedule():
        count = 0
        # Main loop for random schedule
        while (1):
                Eventlist = pygame.event.get()

                newedgeID = random.randint(0,(NUMBEROFVARIABLES - 1) * 2 - 1)
                direction = random.randint(0,1)
                # Which variable and factor does this edge connect to?
                if (newedgeID % 2 == 0):
                        variableID = newedgeID // 2
                        factorID = newedgeID // 2
                else:
                        variableID = (newedgeID + 1) // 2
                        factorID = (newedgeID - 1) // 2
                if (direction == 0):
                        # Variable to factor
                        if (newedgeID % 2 != 0 or factorID == 0):
                                localfactorID = 0
                        else:
                                localfactorID = 1
                        variablenodes[variableID].sendMessage(localfactorID)
                        updateDisplay()
                else:
                        # Factor to variable
                        localvariableID = newedgeID % 2
                        factornodes[factorID].sendMessage(localvariableID)
                count += 1
                print (count)



def floodfillSchedule():

        for variableID in range(NUMBEROFVARIABLES-1):
                if (variableID == 0):
                        variablenodes[variableID].sendMessage(0)
                else:
                        variablenodes[variableID].sendMessage(1)
                factornodes[variableID].sendMessage(1)
                Eventlist = pygame.event.get()
                updateDisplay()
                time.sleep(0.1)
        for variableID in range(NUMBEROFVARIABLES - 1, 0, -1):
                #print (variableID)
                variablenodes[variableID].sendMessage(0)
                factornodes[variableID - 1].sendMessage(0)
                Eventlist = pygame.event.get()
                updateDisplay()
                time.sleep(0.1)
        # Final one so variable 1 gets updated
        variablenodes[0].sendMessage(0)
        updateDisplay()




# Measurement data

measarray = []

measarray.append(np.matrix([[0.3],[ 16.5]]))
measarray.append(np.matrix([[0.4],[ 16.4]]))
measarray.append(np.matrix([[3.4],[ 24.0]]))
measarray.append(np.matrix([[4.5],[ 13.5]]))
measarray.append(np.matrix([[8.0],[ 11.0]]))
measarray.append(np.matrix([[17.1],[ 4.0]]))
measarray.append(np.matrix([[26.2],[ 10.9]]))
measarray.append(np.matrix([[12.3],[ 18.8]]))
measarray.append(np.matrix([[13.2],[ 18.8]]))
measarray.append(np.matrix([[25.9],[ 13.2]]))
measarray.append(np.matrix([[30.4],[ 15.5]]))
measarray.append(np.matrix([[35.3],[ 17.2]]))
        

variablenodes = []
factornodes = []
edges = []




# Initialise variable nodes
for x in range(NUMBEROFVARIABLES):
        variablenodes.append(VariableNode(x, float(x), 0.0))
nextvariableID = NUMBEROFVARIABLES

# Initialise factors: first with smoothness factors
#print ("Initialising smoothness factors")
for newfactorID in range(NUMBEROFVARIABLES - 1):
        # New factor connects this to variable to the right
        #print (newfactorID)
        newfactornode = FactorNode(newfactorID, 0)
        factornodes.append(newfactornode)
        # Set values for smoothness factor
        (eta, Lambdaprime) = SmoothnessFactor(1.0 / SMOOTHNESSsigma ** 2)   # Precision for smoothness
        newfactornode.twovariablefactor.eta += eta
        newfactornode.twovariablefactor.Lambdaprime += Lambdaprime 

# Add on measurement factors
#print ("Measurement factors")
for datapoint in measarray:
        x = datapoint[0,0]
        y = datapoint[1,0]
        factorID = int(x)
        #print ("Datapoint", x, y, "associated with factorID", factorID)
        (eta, Lambdaprime) = MeasurementFactor(variablenodes[factorID].x, variablenodes[factorID + 1].x, y, x, 1.0 / MEASUREMENTsigma ** 2)
        #print ("New factor =", eta, Lambdaprime)
        factornodes[factorID].twovariablefactor.eta += eta
        factornodes[factorID].twovariablefactor.Lambdaprime += Lambdaprime


# Check total factors
#print ("Total factors")
#for factor in factornodes:
        #print ("FactorID", factor.factorID)
        #print (factor.twovariablefactor.eta)
        #print (factor.twovariablefactor.Lambdaprime)


# Connect variables and factors with edges
for newedgeID in range (2 * (NUMBEROFVARIABLES - 1)):
        newedge = Edge(newedgeID)
        edges.append(newedge)
        # Which edge and factor does this edge connect to?
        if (newedgeID % 2 == 0):
                variableID = newedgeID // 2
                factorID = newedgeID // 2
        else:
                variableID = (newedgeID + 1) // 2
                factorID = (newedgeID - 1) // 2
        #print("Connecting new edge", newedgeID, "to variable", variableID, "and factor", factorID)
        variablenodes[variableID].edges.append(newedge)
        factornodes[factorID].edges.append(newedge)










keys_flag = 1


# Keyboard input decides schedule
while (1):
        Eventlist = pygame.event.get()


        # event handling
        for event in Eventlist:
                if (event.type == KEYDOWN):
                        if (event.key == K_f):
                                keys_flag = 0
                                print("Go floodfill schedule.")
                                floodfillSchedule()
                        if (event.key == K_r):
                                keys_flag = 0
                                print("Go random schedule.")
                                randomSchedule()

        updateDisplay()














