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

# add libraries sklearn mutual info for Filter
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

from deap import algorithms # type: ignore
from deap import base # type: ignore
from deap import creator # type: ignore
from deap import tools # type: ignore
from deap.benchmarks.tools import diversity, convergence, hypervolume


def main(data = None,   # wcbd.data - use for feat_select
         NGEN = 25,
         HOF_MAX = 10,
         INDPB = 0.4,
         MU = 50,
         SEL = tools.selNSGA2,
         seed = None):
    
    seed = random.seed(seed)

    ##### read in data and features
    # read in data file
    raw_data = pd.read_csv(data, index_col=False) 
    #(_, ncols) = raw_data.shape
    
    if data == 'musk.data':
      raw_data = raw_data.drop(columns = ['molecule_name', 'conformation_name'])
      
    # build the objects
    X = raw_data.loc[:, raw_data.columns != 'class']
    y = raw_data.loc[:, 'class']
    k = y.nunique()


    
    ##### generic stuff
    IND_INIT_SIZE = len(X.columns) # type: ignore
    LAMBDA = MU

    
    # remember individuals that have previously been evaluated
    rememberer = []

    # NSGA2 evaluator
    def evalNSGA2(individual):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        
        # check if duplicate individual has already been evaluated
        if individual in rememberer:
          # retrieve the previously evaluated fitness.values
          (error, feat_ratio) = rememberer[rememberer.index(individual)].fitness.values
          
          return error, feat_ratio
        
        elif sum(individual) == 0:
          error, feat_ratio = 1, 1
          individual.fitness.values = (error, feat_ratio)
          rememberer.append(individual)
          
          return error, feat_ratio

        else:
          
          selected = []
          
          # identify selected features
          for i, feat in enumerate(individual):
            if feat != 0:
              selected.append(i)
          
          # build selected X and y
          X_eval = X.iloc[:, selected]
          
          # split for train test to get accuracy
          X_train, X_test, y_train, y_test = train_test_split(X_eval, y, test_size = 0.3)

          feat_ratio = len(selected)/len(individual)
          knn = KNeighborsClassifier(k)
          knn.fit(X_train, y_train)
          error = 1- knn.score(X_test, y_test) # error = 1-accuracy
          
          individual.fitness.values = (error, feat_ratio)
          rememberer.append(individual)
  
          return error, feat_ratio
        
    # get rid of that annoying warning 
    try:
      del creator.FitnessNSGA
    except Exception as e:
      pass
    
    try:
      del creator.Individual_NSGA
    except Exception as e:
      pass


    # Individual representation:
    # minimise error; minimise length
    creator.create("FitnessNSGA", base.Fitness, weights=(-1.0, -1.0 ))
    creator.create("Individual_NSGA", list, fitness=creator.FitnessNSGA) # type: ignore


    # Tools in the toolbox
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1) 
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual_NSGA,  # type: ignore
        toolbox.attr_bool, IND_INIT_SIZE) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore



    # evaluate, mate, mutate, select, hall of fame, population
    toolbox.register('evaluate', evalNSGA2)
    toolbox.register("mate", tools.cxOnePoint) #, indpb = INDPB)  
    toolbox.register("mutate", tools.mutFlipBit, indpb = INDPB)#, low = LOW, up = UP)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("hof", tools.HallOfFame)
    
    pop = toolbox.population(n=MU) # type: ignore
    hof = toolbox.hof(1) # type: ignore

    # statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
        
    # run the genetic algorithm
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, cxpb=0.6, mutpb=0.4, ngen=NGEN, stats=stats,
                              halloffame=hof, verbose = False)
    hyp = hypervolume(pop, [1.01, 1.01])
    
    return pop, logbook , hof, hyp

if __name__ == "__main__":
    main()                 
