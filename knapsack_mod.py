#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import pandas as pd # type: ignore
import numpy # type: ignore
import itertools

from deap import algorithms # type: ignore
from deap import base # type: ignore
from deap import creator # type: ignore
from deap import tools # type: ignore


def main(data = "10_269", 
         NGEN = 50,
         HOF_MAX = 10,
         INDPB = 0.4,
         SEL = tools.selBest,
         EMPTY = 0.9,
         CX = tools.cxUniform,
         CXPB = 0.6,
         MUTPB = 0.4,
         ALGO = algorithms.eaMuPlusLambda):
    
    items = pd.read_csv(data, sep = ' ')
    MAX_ITEM, MAX_WEIGHT = [int(i) for i in items.columns]
    items.columns = ['value', 'weight']
    items = items.to_dict(orient = 'records')
    
    NBR_ITEMS = MAX_ITEM
    IND_INIT_SIZE = MAX_ITEM
    GEN_1 = 5 * NBR_ITEMS
    MU = GEN_1
    EMPTY_BIAS = []
    for i in range(IND_INIT_SIZE):
        EMPTY_BIAS.append(0) if i < (int(EMPTY * IND_INIT_SIZE)) else EMPTY_BIAS.append(1)
    LAMBDA = 2*GEN_1
    NGEN = 50
    TOURN_SIZE = 10
    LOW = 0
    UP = 1
    
    
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0 ))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.choice, EMPTY_BIAS) # change from random.randrange to select binary, no count
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
        toolbox.attr_bool, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalKnapsack(individual):
        weight = 0.0
        value = 0.0
        for i, item in enumerate(individual):
            weight += (item * items[i]['weight'])
            value += (item * items[i]['value'])
        if weight > MAX_WEIGHT:
            return 0, 10000  # Ensure overweighted bags are dominated 

        return value, weight    
    
    def topfive(pop):
        mean_val = numpy.mean(sorted([ind[0] for ind in pop], reverse = True)[:5])
        mean_weight = numpy.mean(sorted([ind[1] for ind in pop], reverse = True)[:5])
        
        return mean_val , mean_weight

    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", CX, indpb = INDPB)  
    toolbox.register("mutate", tools.mutFlipBit, indpb = INDPB)#, low = LOW, up = UP)
    toolbox.register("select", SEL)
    toolbox.register("hof", tools.HallOfFame)
    
    pop = toolbox.population(n=GEN_1)
    hof = toolbox.hof(HOF_MAX)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    # stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    stats.register("top", topfive)
    
    pop, logbook = ALGO(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose = False)
    
    return pop, logbook , hof 

if __name__ == "__main__":
    main()                 