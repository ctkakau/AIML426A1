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


def main(data = None,   # wcbd.data - use for feat_select
         feature_names = None,   #wbcd.names - use for feat_select
         FEAT_SEL = 'Filter',   # default fitness function is Filter
         clf = KNeighborsClassifier,  # default classifier KNN
         NGEN = 50,
         HOF_MAX = 10,
         INDPB = 0.4,
         SEL = tools.selBest,
         CX = tools.cxUniform,
         CXPB = 0.6,
         MUTPB = 0.4,
         ALGO = algorithms.eaMuPlusLambda):
    

    # read in name file

    feat_names = pd.read_csv(feature_names, sep = ":", skiprows = [0], names = ['feature', 'type'])
    NBR_FEATS = len(feat_names)

    feat_names.loc[NBR_FEATS+1, ['feature', 'type']] = ['class', 'discrete']
    # read in raw data    
    raw_data = pd.read_csv(data, index_col=False, names = [feat for feat in feat_names.loc[:, 'feature']])

    IND_INIT_SIZE = NBR_FEATS
    MU = NBR_FEATS
    LAMBDA = MU
    NGEN = 50

    # remove these if no tournament
    TOURN_SIZE = 10
    LOW = 0
    UP = 1
    
    
    # Individual representation:
    # maximise mutual info gain (Filter) or AIC(wrapper); minimise length
    creator.create("FitnessFS", base.Fitness, weights=(1.0, -1.0 ))
    creator.create("Individual_FS", list, fitness=creator.FitnessFS)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1) 

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual_FS, 
        toolbox.attr_bool, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Filter feature selection
    def evalFilterGA(individual):
        selected = []
        # identify selected features
        for i, feat in enumerate(individual):
            
            if feat != 0:
                selected.append(i)
        
        # build selected X and y
        X = raw_data.iloc[:, selected]
        y = raw_data.loc[:, 'class']

        # compute the average mutual info of reduced features per feature
        length = len(X.columns)
        avg_MI_gain = sum(mutual_info_classif(X, y))/length

        return avg_MI_gain, length
    
    # Wrapper selection
    def evalWrapperGA(individual):
        selected = []
        # identify selected features
        for i, feat in enumerate(individual):
            
            if feat != 0:
                selected.append(i)
        
        # build selected X and y
        X = raw_data.iloc[:, selected]
        y = raw_data.loc[:, 'class']

        # compute classification accuracy based on some classification model
        # sklearn K nearest neighbours
        length = len(X.columns)
        knn = KNeighborsClassifier(2)
        knn.fit(X, y)
        accuracy = knn.score(X, y)

        return accuracy, length
    
    if FEAT_SEL == 'Filter':
        toolbox.register("evaluate", evalFilterGA)
    else:
        toolbox.register('evaluate', evalWrapperGA)
    toolbox.register("mate", CX, indpb = INDPB)  
    toolbox.register("mutate", tools.mutFlipBit, indpb = INDPB)#, low = LOW, up = UP)
    toolbox.register("select", SEL)
    toolbox.register("hof", tools.HallOfFame)
    
    pop = toolbox.population(n=MU)
    hof = toolbox.hof(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max, axis=0)
        
    pop, logbook = ALGO(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose = False)
    
    return pop, logbook , hof 

if __name__ == "__main__":
    main()                 