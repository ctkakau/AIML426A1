#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def _div_(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
      
def _if_(input, output1, output2):
  return output1 if input else output2

pset = gp.PrimitiveSetTyped("MAIN", [float,], float)
# boolean primitives
pset.addPrimitive(operator.xor, [bool, bool], bool) 
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)


# boolean float primitives
pset.addPrimitive(_if_, [bool, float, float], float)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)


# float primitives
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(_div_, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)

pset.addEphemeralConstant('rand101', partial(random.choice, [-1.0, 0.0,  1.0] ), float) 

# float and boolean terminals
# pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float) # add float terminal
pset.addTerminal(2.0, float) # add float terminal
pset.addTerminal(3.0, float)
pset.addTerminal(5.0, float)
pset.addTerminal(7.0, float)
pset.addTerminal(1, bool)
# pset.addTerminal(0, bool)

pset.renameArguments(ARG0='x')


creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # f(x) = 1/x +sinx, for x>o ; 2x+x**2 + 3.0, for x <= 0
    sqerrors = ((func(x) - 1/x - math.sin(x))**2 if (x>0) else (func(x)- 2*x - x**2 - 3.0)**2 for x in points)
    
    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-3,3)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint,) #LeafBiased, termpb = 0.3,) # prob of selecting terminal
toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # from mutUniform

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5)) # 17 these trees are ridiculous
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

def main(seed = 318):
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=False)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()
