from deap import base, creator
import random
import numpy as np
from deap import tools
import fitness_function as ff
from dataset.test import FeatureSelection
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import pandas as pd

class FeatureSelectionGA(FeatureSelection):
    """
        FeaturesSelectionGA
        This class uses Genetic Algorithm to find out the best features for an input model
        using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
        used for GA but it can be changed accordingly.


    """
    # def __init__(self,model, x, y, cv_split=5, verbose=0):
    def __init__(self, csv, model = svm.SVC(), cv_split=5, verbose=0, num_feature_select = 10):

        """
            Parameters:

            model : scikit-learn supported model,
                x :  {array-like}, shape = [n_samples, n_features]
                     Training vectors, where n_samples is the number of samples
                     and n_features is the number of features.

                y  : {array-like}, shape = [n_samples]
                     Target Values
            cv_split: int
                     Number of splits for cross_validation to calculate fitness.

            verbose: 0 or 1
        """

        super().__init__(csv, num_feature_select)

        self.mutual_info_calculator()

        self.model =  model
        self.csv_data = pd.read_csv(csv)
        self.x = self.csv_data.iloc[:, self.top_n_features]
        # self.x = self.csv_data.iloc[:, 0:14]
        self.y = self.csv_data.iloc[:, 14: 15].values.reshape(-1,)


        self.n_features = self.x.shape[1]
        self.toolbox = None
        self.creator = self._create()
        self.cv_split = cv_split

        self.verbose = verbose

        if self.verbose==1:
            print("Model {} will select best features among {} features using cv_split :{}.".format(model,x.shape[1],cv_split))
            print("Shape od train_x: {} and target: {}".format(x.shape,y.shape))

        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None

    def evaluate(self,individual):
        fit_obj = ff.FitenessFunction(self.cv_split)
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind==1)[0]
            fitness = fit_obj.calculate_fitness(self.model,self.x.iloc[:,feature_idx],self.y)

        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual,fitness))

        return fitness,


    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator

    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox


    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        return toolbox

    def get_final_scores(self,pop,fits):
        self.final_fitness = list(zip(pop,fits))



    def generate(self,n_pop,cxpb = 0.5,mutxpb = 0.2,ngen=5,set_toolbox = False):

        """
            Generate evolved population
            Parameters
            -----------
                n_pop : {int}
                        population size
                cxpb  : {float}
                        crossover probablity
                mutxpb: {float}
                        mutation probablity
                n_gen : {int}
                        number of generations
                set_toolbox : {boolean}
                              If True then you have to create custom toolbox before calling
                              method. If False use default toolbox.
            Returns
            --------
                Fittest population
        """



        if self.verbose==1:
            print("Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(n_pop,cxpb,mutxpb,ngen))

        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception("Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox")
        pop = self.toolbox.population(n_pop)
        CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

        # print(pop)


        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # print(ind, ind.fitness, ind.fitness.values)


        for g in range(NGEN):
            print("-- GENERATION {} --".format(g+1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in pop])
            # Clone the selected individuals

            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring

            # print(offspring)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        # if self.verbose==1:
        #     print("  Min %s" % min(fits))
        #     print("  Max %s" % max(fits))
        #     print("  Avg %s" % mean)
        #     print("  Std %s" % std)

        print("-- Only the fittest survives --")

        self.best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        self.get_final_scores(pop,fits)

        return pop

    def train(self):
        fit_obj = ff.FitenessFunction(10)
        feature_idx = np.where(np.asarray(self.best_ind)==1)[0]
        # print(feature_idx, self.best_ind)
        fitness = fit_obj.calculate_fitness(self.model,self.x.iloc[:,feature_idx],self.y)
        print("The accuracy using feature set {} is {}%".format(feature_idx,fitness * 100))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

f = FeatureSelectionGA("australian.csv", clf)
f.generate(60, ngen = 10)
f.train()
