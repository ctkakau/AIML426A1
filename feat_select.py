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
    

  
    ##### read in data and features
    
    # read in name file
    feat_names = pd.read_csv(feature_names, sep = ":", skiprows = [0], names = ['feature', 'type']) # type: ignore
    NBR_FEATS = len(feat_names)
    feat_names.loc[NBR_FEATS+1, ['feature', 'type']] = ['class', 'discrete']
    # read in raw data    
    raw_data = pd.read_csv(data, index_col=False, names = [feat for feat in feat_names.loc[:, 'feature']]) # type: ignore


    # function to discretize data
    def discretize_data(X):
        import pandas as pd
        
        X_disc = pd.DataFrame(columns = X.columns, index = X.index)
        
        # cut each column into 10 mins
        for column, _ in enumerate(X.columns): 
            X_disc.iloc[:, column] = pd.cut(X.iloc[:, column], bins = 10,)
            
            # keep the value for the left interval
            for row, _ in enumerate(X_disc.iloc[:, column]):
                X_disc.iloc[row, column] = X_disc.iloc[row, column].left # type: ignore
            
        return X_disc
    
    

    # functions to compute individual and joint entropies
    # create a function to calculate probabilities of discretised vectors
    def joint_probabilities(X_i, y):
        import pandas as pd
        import numpy as np
        
        # establish classes
        classes_of_y = y.unique()
        levels_of_X_i = X_i.unique()
        
        # establish output object
        probs = pd.DataFrame(index = levels_of_X_i, columns = classes_of_y, )
        # establish probabilities for each category of x
        # counts
        for clas in classes_of_y:
            count = pd.Series(X_i[y==clas]).value_counts()
            prob = count/len(X_i)
            probs[clas] = prob
            
        return probs
        

   # create a function to calculate entropy
    def entropy(probabilities):
        import pandas as pd
        import numpy as np
        
        entropy = 0
        
        # allow for single columns
        if isinstance(probabilities, pd.Series):
            ent = pd.DataFrame(columns = probabilities.index)
            ent = - np.sum(probabilities* np.log2(probabilities))
            
            entropy = ent
        
        else:
            # compute Shannon's entropy for 
            for prob in probabilities.columns:
            
                ent = - np.sum(probabilities.loc[:, prob]* np.log2(probabilities.loc[:, prob]))
                
                entropy += ent
            
        entropy = -entropy  
        return entropy
    

    # function to calculate mutual information
    def mutual_information(features, joint = None, indiv = None, clas = None):
        import numpy as np
        
        # features is an array of indexes for features
        H_X = 0
        H_XY = 0
        H_Y = clas
        for feature in features:
            H_X += indiv[feature]
            H_XY += joint[feature]
            
        MI = H_X +H_Y - H_XY
        
        return MI


    ##### establish X and y
    X = raw_data.iloc[:, :NBR_FEATS]
    discretized_X = discretize_data(X)
    y = raw_data['class']

    # compute joint entropies for each individual feature
    joint_entropies = [entropy(joint_probabilities(discretized_X.loc[:, i], y)) for i in discretized_X.columns]
    individual_entropies = [entropy(pd.Series(discretized_X.loc[:, i]).value_counts()/len(y)) for i in discretized_X.columns]
    class_entropy = entropy(pd.Series(y).value_counts()/len(y))


    ##### generic stuff
    IND_INIT_SIZE = NBR_FEATS # type: ignore
    MU = NBR_FEATS # type: ignore
    LAMBDA = MU



    # Filter feature selection
    def evalFilterGA(individual):
        from sklearn.feature_selection import mutual_info_classif
        
        selected = []
        

        # identify selected features
        for i, feat in enumerate(individual):
           
            if feat != 0:
                selected.append(i)
        
        MI_gain = mutual_information(features = selected, joint = joint_entropies, indiv = individual_entropies, clas = class_entropy)
        
        # average over total features selected
        length = len(X.columns)
        avg_MI_gain = MI_gain/length

        return avg_MI_gain, length
    


    # Wrapper selection
    def evalWrapperGA(individual):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        
        selected = []
        
        # identify selected features
        for i, feat in enumerate(individual):
            
            if feat != 0:
                selected.append(i)
        
        # build selected X and y
        X = raw_data.iloc[:, selected]
        y = raw_data.loc[:, 'class']

        # split for train test to get accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

        # compute classification accuracy based on some classification model
        # sklearn K nearest neighbours
        length = len(X.columns)
        knn = KNeighborsClassifier(2)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)

        return accuracy, length




    # Individual representation:
    # maximise mutual info gain (Filter) or AIC(wrapper); minimise length
    creator.create("FitnessFS", base.Fitness, weights=(1.0, -1.0 ))
    creator.create("Individual_FS", list, fitness=creator.FitnessFS) # type: ignore


    # Tools in the toolbox
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1) 
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual_FS,  # type: ignore
        toolbox.attr_bool, IND_INIT_SIZE) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore



    # evaluate, mate, mutate, select, hall of fame, population
    if FEAT_SEL == 'Filter':
        toolbox.register("evaluate", evalFilterGA)
    else:
        toolbox.register('evaluate', evalWrapperGA)
    toolbox.register("mate", CX, indpb = INDPB)  
    toolbox.register("mutate", tools.mutFlipBit, indpb = INDPB)#, low = LOW, up = UP)
    toolbox.register("select", SEL)
    toolbox.register("hof", tools.HallOfFame)
    
    pop = toolbox.population(n=MU) # type: ignore
    hof = toolbox.hof(1) # type: ignore

    # statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max, axis=0)
        
    # run the genetic algorithm
    pop, logbook = ALGO(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose = False)
    
    return pop, logbook , hof 

if __name__ == "__main__":
    main()                 
