import numpy as np
import scipy.integrate 
from enum import IntEnum, IntFlag, auto
import matplotlib.pyplot as plt
import collections


class State(IntEnum):
    Susceptible = 0
    Infected = 1
    Asymptomatic = 2
    Symptomatic = 3
    RequiresHospitalization = 4
    Recovered = 5
    Dead = 6
    Quarantined = 7

class MeasureType(IntFlag):
    Quarantine = auto()
    QuarantineEffectiveness = auto()
    SocialDistancing = auto()
    SocialAwareness = auto()

class TriggerType(IntFlag):
    Timed = auto()
    SickProportion = auto()
    SymptomaticProportion = auto()
    DeathProportion = auto()
    BiggerThan = auto()
    SmallerThan = auto()

class Disease:
    def __init__(self, transmissionRate, incubationLength, symptomsProbability, sicknessLength, criticalCareRequiredProbability, lethality):
        self.transmissionRate = transmissionRate
        self.incubationLength = incubationLength
        self.symptomsProbability = symptomsProbability
        self.sicknessLength = sicknessLength
        self.criticalCareRequiredProbability = criticalCareRequiredProbability
        self.lethality = lethality

    def getDerivs(self, t, y, dy, infectiousPpl, interactionRate):
        R = interactionRate * self.transmissionRate

        dy[State.Susceptible] -= infectiousPpl * R * y[State.Susceptible]
        dy[State.Infected]    +=  infectiousPpl * R * y[State.Susceptible] 
        
        dy[State.Infected]     -= 1./self.incubationLength * y[State.Infected]
        dy[State.Symptomatic]  += self.symptomsProbability * 1./self.incubationLength * y[State.Infected]
        dy[State.Asymptomatic] += (1.-self.symptomsProbability) * 1./self.incubationLength * y[State.Infected]
        
        dy[State.Symptomatic]  -= (1.-self.criticalCareRequiredProbability)/self.sicknessLength * y[State.Symptomatic]
        dy[State.Asymptomatic] -= 1./self.sicknessLength * y[State.Asymptomatic]
        dy[State.Recovered]    += 1./self.sicknessLength * ((1.-self.criticalCareRequiredProbability) * y[State.Symptomatic] + y[State.Asymptomatic])

        dy[State.Symptomatic]             -= self.criticalCareRequiredProbability/self.sicknessLength * y[State.Symptomatic]
        dy[State.RequiresHospitalization] += self.criticalCareRequiredProbability/self.sicknessLength * y[State.Symptomatic]

        dy[State.RequiresHospitalization] -= self.lethality/self.sicknessLength * y[State.RequiresHospitalization]
        dy[State.Dead]                    += self.lethality/self.sicknessLength * y[State.RequiresHospitalization]
        return R


class Measure:
    def __init__(self, measureType, measureVal, triggerType, triggerVal=0., triggerTime=0., name = 'measure'):
        self.implemented = False
        self.type = measureType
        self.measureVal = measureVal
        self.triggerType = triggerType
        self.triggerVal = triggerVal
        self.triggerTime = triggerTime
        self.name = name

    def checkTrigger(self, t, y, community, disease):
        if self.implemented:
            return False
        if TriggerType.Timed in self.triggerType:
            if t < self.triggerTime:
                return False
        comp = 0
        if TriggerType.SickProportion in self.triggerType:
            comp = y[State.Asymptomatic] + y[State.Symptomatic] + y[State.RequiresHospitalization]
        elif TriggerType.SymptomaticProportion in self.triggerType:
            comp = y[State.Symptomatic] 
        else:
            comp = y[State.Dead]

        if TriggerType.BiggerThan in self.triggerType:
            if comp < self.triggerVal:
                return False
        elif TriggerType.SmallerThan in self.triggerType:
            if comp > self.triggerVal:
                return False
        return True

    def implementSingle(self, community, t, y, type, val):
        if MeasureType.SocialDistancing in type:
            community.socialDistancing = val
        elif MeasureType.Quarantine in type:
            community.quarantineMeasures = val
        elif MeasureType.SocialAwareness in type:
            community.socialAwareness = val
        elif MeasureType.QuarantineEffectiveness in type:
            community.QuarantineEffectiveness = val

    def implement(self, community, t, y):
        self.t = t
        self.implemented = True
        if isinstance(self.type, collections.abc.Sequence):
            for i in range(len(self.type)):
                self.implementSingle(community, t, y, self.type[i], self.measureVal[i])
        else:
            self.implementSingle(community, t, y, self.type, self.measureVal)


class Community:
    def __init__(self, population, popInteractionRate, maxCareCapacity, name="community"):
        self.population = population
        self.popInteractionRate = popInteractionRate
        self.maxCareCapacity = maxCareCapacity
        self.socialAwareness = 0.
        self.quarantineMeasures = 0
        self.quarantineEffectiveness = 1.
        self.socialDistancing = 0.
        self.name = name
        self.measures = []
        self.hist = []

    def reset(self):
        self.hist = []
        for m in self.measures:
            m.implemented = False

    # calculates current Paramters important for the disease
    # this is an arbitrary model, popInteractionRate could correspond to the number of people an individual is in contact with each day
    # the infectiousPpl are responsible for spreading the disease, socialAwareness could correspond to those people self isolating and everyones Hygiene
    # careCapacity is reduced when doctors are sick in this model
    def getCurrentParameters(self, t, y, disease):
        interactionRate = self.popInteractionRate*(1. - self.socialDistancing) 
        careCapacity =  self.maxCareCapacity * (1. - y[State.Symptomatic] - y[State.RequiresHospitalization] - y[State.Dead] - np.sum(y[State.Quarantined:]))
        infectiousPpl = (1. - self.socialAwareness/2.) * y[State.Asymptomatic] + (1. - self.socialAwareness) * y[State.Symptomatic] 
                            # + np.max([1. - 5.*self.socialAwareness, 0.])* y[State.RequiresHospitalization]
        self.hist.append([t, interactionRate, careCapacity, infectiousPpl])
        return interactionRate, careCapacity, infectiousPpl

    
    def checkForMeasures(self, t, y, disease):
        for m in self.measures:
            if m.checkTrigger(t, y, self, disease):
                m.implement(self, t, y)

    # Quarantine Measures are implemented such that the vector functions 
    # [State.Susceptible, State.Asymptomtic, ..., State.Quarantined + State.Susceptible, State.Asymptomatic]
    # This is the only function where the interplay between the two parts of the vector is calculated
    # The disease class only sees either part and calculates the natural progression
    def quarantineDerivs(self, t, y, dy, disease):
        disease.getDerivs(t, y[State.Quarantined:], dy[State.Quarantined:], 0., 0)

        dy[State.Quarantined + State.RequiresHospitalization] -= y[State.Quarantined + State.RequiresHospitalization]
        dy[State.RequiresHospitalization]                     += y[State.Quarantined + State.RequiresHospitalization]

        dy[State.Quarantined + State.Dead] -= y[State.Quarantined + State.Dead]
        dy[State.Dead]                     += y[State.Quarantined + State.Dead]

        dy[State.Quarantined + State.Recovered] -= y[State.Quarantined + State.Recovered]
        dy[State.Recovered]                     += y[State.Quarantined + State.Recovered]
        
        if self.quarantineMeasures > 0:
            dy[State.Quarantined + State.Symptomatic] += self.quarantineEffectiveness * y[State.Symptomatic]
            dy[State.Symptomatic]                     -= self.quarantineEffectiveness * y[State.Symptomatic]

        if self.quarantineMeasures > 1:
            dy[State.Quarantined + State.Asymptomatic] += self.quarantineEffectiveness * y[State.Asymptomatic]
            dy[State.Asymptomatic]                     -= self.quarantineEffectiveness * y[State.Asymptomatic]

        if self.quarantineMeasures > 2:
            dy[State.Quarantined + State.Infected] += self.quarantineEffectiveness * y[State.Infected]
            dy[State.Infected]                     -= self.quarantineEffectiveness * y[State.Infected]


    # This function first hands off to the Disease class to calculate the natural procession of it
    # and then adds community effects to it, for example careCapacity and quarantineMeasures
    def getDerivs(self, t, y, dy, disease):
        self.checkForMeasures(t, y, disease)
        interactionRate, careCapacity, infectiousPpl = self.getCurrentParameters(t, y, disease)
        
        R = disease.getDerivs(t, y, dy, infectiousPpl, interactionRate)
        self.quarantineDerivs(t, y, dy,  disease)
         
        # if there is not enough care Capacity additional people will die
        dy[State.RequiresHospitalization] -= np.max([y[State.RequiresHospitalization] - careCapacity, 0.])
        dy[State.Dead]                    += np.max([y[State.RequiresHospitalization] - careCapacity, 0.])

        dy[State.RequiresHospitalization] -= (1.-disease.lethality)/disease.sicknessLength * np.min([y[State.RequiresHospitalization], careCapacity])
        dy[State.Recovered]               += (1.-disease.lethality)/disease.sicknessLength * np.min([y[State.RequiresHospitalization], careCapacity])
        # print( t,y, dy, R, careCapacity)
        self.hist[-1][1] = R

        

# This function is called in every step and delegates to the communities
def derivs(y, t, disease, community):
    dy = np.zeros(np.shape(y))
    community.getDerivs(t, y, dy, disease)
    # dy = np.round(dy*community.population) / community.population  # tried to atomize calculation - did not wok
    # print(t, y, dy)
    return dy


def initialConditions(community):
    y0 = np.zeros(len(State)*2)  # This is doubled to accomodate for quarantine states
    y0[State.Susceptible] = 1. - 1./community.population    # Only one person infected
    y0[State.Infected] = 1./community.population
    return y0

def simulate(disease, community, y0, tfin):
    community.reset()
    t = np.linspace(0, tfin, num=int(tfin)*2, endpoint=True)
    print(y0,)
    ys, infodict = scipy.integrate.odeint(derivs, y0, t, args=(disease,community), full_output=True, printmessg=True)
    # print(infodict)
    return t, ys

def plotCommunity(ax, community, t, ys, disease):
    ax.set_title(community.name)
    ax.set_xlabel("Time / days")
    ax.set_ylabel("Population")
    ax.plot(t, [y[State.Susceptible] for y in ys], label='susceptible', color='yellow')
    ax.plot(t, [y[State.Asymptomatic] + y[State.Symptomatic]+ y[State.RequiresHospitalization]
                    + y[State.Quarantined + State.Asymptomatic] + y[State.Quarantined + State.Symptomatic]for y in ys] , label='sick', color='orange')
    ax.plot(t, [y[State.RequiresHospitalization] for y in ys], label='hospitalization', color='red')
    ax.plot(t, [y[State.Recovered] for y in ys], label='recovered', color='green')
    ax.plot(t, [y[State.Dead] for y in ys], label='dead', color='black')
    ax.plot(t, [np.sum(y[State.Quarantined:]) for y in ys], label='quarantined', color='blue')
    ax.plot([y[0] for y in community.hist], [y[2] for y in community.hist], label='care capacity', linestyle='--', color='red')
    ax.plot([y[0] for y in community.hist], [y[1]*1e-1 for y in community.hist], label='R/10', linestyle='--', color='orange')
    ax.grid(); ax.set_yscale('log'); ax.set_ylim(bottom=1e-6)
    ax.legend()
    for m in community.measures:
        if m.implemented:
            ax.axvline(m.t, 0., 1., color = 'yellow', linestyle='--')
            ax.text(m.t , 0.1, m.name, color= 'yellow', fontsize = 8, rotation=90)



def main():
    covid = Disease(0.7, 3., 0.5, 14., 0.3, 0.2)
    aachen = Community(250000.,5., 0.03, name='aachen')
    aachen.measures.append(Measure( [MeasureType.SocialDistancing, MeasureType.Quarantine, MeasureType.SocialAwareness], [0.5, 1., 0.5], 
                                        TriggerType.Timed | TriggerType.SickProportion | TriggerType.BiggerThan,  triggerVal=.0001, triggerTime=2.))
    aachen.measures.append(Measure( [MeasureType.SocialDistancing, MeasureType.Quarantine, MeasureType.SocialAwareness], [0.96, 2., 1.], 
                                        TriggerType.Timed | TriggerType.SickProportion | TriggerType.BiggerThan,  triggerVal=.01, triggerTime=7.))
    aachen.measures.append(Measure([MeasureType.SocialDistancing, MeasureType.Quarantine], [0.5, 1.],TriggerType.Timed | TriggerType.SickProportion | TriggerType.SmallerThan,  triggerVal=.0001, triggerTime=100.))

    y0 = initialConditions(aachen)
    tfin = 500
    t, ys = simulate(covid, aachen, y0, tfin)
    # print(t, ys)
    plotCommunity(plt.gca(), aachen, t, ys, covid)
    plt.show()

if __name__ == "__main__":
    main()




