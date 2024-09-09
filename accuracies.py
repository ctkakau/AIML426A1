# a bunch of functions for getting the best individual
# in a population; selecting features; modelling and checking accuracies


# function to retun the best individual in a population based on fitness
def best_in_population(pop, fitness = 'fitness'):
  """determine the best individual in an population based on fitness"""
  from operator import attrgetter
  
  # retrieve from the population, the individual with the best fitness
  best = max(pop, key = attrgetter(fitness))
  
  return best



# function to extract X based on best individual's feature selection
def selected_features(best, X):
  """selects the features from an idividual identified as the best
  and outputs a reduced dataset using those selected features
  """
  
  import pandas as pd
  
  selected = []
  
  # create a set of selected features based on the chromosomes of the best individual
  for f, feature in enumerate(best):
    
    # retain the selected features
    if feature == 1:
      selected.append(f)
    
  # generate the reduced X with just selected features
  reduced_X = X.iloc[:, selected]
  
  return reduced_X



# function to determine training accuracy of a dataset using DecisionTreeClassifier
def accuracy_of_selected_features(X, y):
  """determine the accuracy of a dataset"""
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  
  # split data for training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
  
  # run the classifier
  clf = DecisionTreeClassifier()
  clf = clf.fit(X_train, y_train)
  
  accuracy = clf.score(X_test, y_test)
  
  return accuracy


# function for running experiment N-times
def multi_run(func, data = None, feat_names = None, params = None, n=1, seed = None):
    """create multiple runs of a selected experiment"""
    from datetime import datetime as dt
    # accept only singular parameters - i.e. feed one function only
    hallOfFame = []
    run_stats = []
    time_delta = []
    population = []

    for r in range(n): 
        start = dt.now()
        if feat_names:
            if params: # feat_sel_GA
                pop, stat, hof = func(data, feat_names, params)
                
            else: # knapsack
                pop, stat, hof = func(data, feat_names)
                
        else: #symbreg
          if data is not None:
            pop, stat, hof, hyp = func(data, seed = r)
          else:
            pop, stat, hof = func(seed = r)
          
        
        end = dt.now()
        time = end - start

        run_stats.append(stat)
        hallOfFame.append(hof)
        time_delta.append(time)
        population.append(pop)
                
    return population, run_stats, hallOfFame, time_delta, hyp


# function for running through the different datasets or different functions
def changing_run(paths = None, feat_names = None, func = None, params = None, n = 1):
  """run an experiment over different datasets"""

  diffs = [[], [], [], []] 

  if len(paths) > 1:
      # iterate paths, keep all other parameters the same, function is required, params are optional
      for p, dat in enumerate(paths):
          if feat_names is not None:
              # include feat_names input
              if params:
                  run, best, time, population = multi_run(func, 
                                         data= dat, 
                                         feat_names= feat_names[p], 
                                         params = params, 
                                         n = n)
              else:
                  run, best, time, population = multi_run(func, 
                                         data= dat, 
                                         feat_names= feat_names[p], 
                                         n = n)
          
          else:
              
              if params:
                  run, best, time, population = multi_run(func, 
                                         data= dat, 
                                         params = params, 
                                         n = n)
              else:
                  run, best, time, population = multi_run(func, 
                                         data= dat, 
                                         n = n)
          diffs[0].append(run)
          diffs[1].append(best)
          diffs[2].append(time)
          diffs[3].append(population)

  return diffs


# function to convert results into a table
def results_table(populations):
  import pandas as pd
  import numpy as np
  import deap
  
  best_indivs = [best_in_population(pop) for pop in populations]
  
  fitness = [round(indiv.fitness.values[0], 3) for indiv in best_indivs]
  programme_size = [round(indiv.fitness.values[1], 2) for indiv in best_indivs]
  stats = pd.DataFrame({'fitness_error' : [np.mean(fitness), np.std(fitness)] , 'fitness_features': [np.mean(programme_size), np.std(programme_size)]}, index = ['average', 'std_dev'])
  results = pd.DataFrame({'fitness_error' : fitness, 'fitness_features': programme_size}, index = ['run'+str(i+1) for i in range(len(fitness))])
  
  results = pd.concat([results, stats])

  return results



# use networkx to draw a tree of a gp individual
def draw_tree(indiv):
  from deap import gp
  import networkx as nx
  import matplotlib.pyplot as plt
  
  # use deap.gp.graph function to collect tree info
  nodes, edges, labels = gp.graph(indiv)
  
  g = nx.Graph()
  
  g.add_nodes_from(nodes)
  g.add_edges_from(edges)
  pos = nx.nx_agraph.graphviz_layout(g, prog = 'dot')
  
  # will this help with over-writes?
  # H = nx.convert_node_labels_to_integers(g, label_attribute="node_label")
  # 
  # H_layout = nx.nx_agraph.pygraphviz_layout(g, prog="dot")
  # 
  # G_layout = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

  nx.draw_networkx_nodes(g, pos, node_size = 300)
  nx.draw_networkx_edges(g, pos)
  nx.draw_networkx_labels(g, pos, labels)
  
  plt.show()


# function to draw a tree using pygraphviz
def draw_pgv_tree(indiv, path = None):
  from deap import gp
  import pygraphviz as pgv
  
  nodes, edges, labels = gp.graph(indiv)

  g = pgv.AGraph()
  g.add_nodes_from(nodes)
  g.add_edges_from(edges)
  g.layout(prog = 'dot')
  
  for i in nodes:
    n = g.get_node(i)
    n.attr['label'] = labels[i]
    
  g.draw() if path is None else g.draw(path)
