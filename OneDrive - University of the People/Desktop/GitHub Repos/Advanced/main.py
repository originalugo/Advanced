''' GENETIC ALGORITHM IMPLEMENTATION '''


''' Importing necessary libraries '''
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Just wanted to time my code
start_time = time.time()

''' ### FUNCTION DEFINITIONS #### '''

def import_data() -> list[dict]: 
    ''' This function imports and cleans the original data to a desired format'''
    
    # The path to the file containing the data

    path = r'C:\Users\Liz Egbuchulam\OneDrive - University of Exeter\Academic Courses\ECMM409 - Nature Inspired Computation\Course_work\ca1.txt'
    
    bags = {}
    raw_file = pd.read_csv(path,  header=None)
    file_dict = raw_file.drop(index=0).to_dict()[0]
    i = 1
    bag_list = []

    while i <= len(file_dict)-2: 
        bags = {
        file_dict[i]: {file_dict[i+1], file_dict[i+2]}
    }
        i +=3
        bag_list.append(bags)


    tmp_list = []
    for i in range(0,len(bag_list)):
        for item in list(bag_list[i].values()):
            for ele in item:
                p = ele.strip(r'\w').strip().split(':')
                tmp_list.append(p)

    tmp_list2 = [float((ele[1].strip())) for ele in tmp_list]


    weights = []; values = []
    for num in tmp_list2:
        if num<10:
            weights.append(num)
        else:
            values.append(num)

    dict = {}
    given_population = []
    for i in range(0,100):
        dict = {
        'Bag ' + str(i+1): {
            'weight': weights[i], 
            'value': values[i]
            }
    }
        given_population.append(dict)
    
    return given_population

def fitness_function(solution) -> list[float,float,float]: # the arg is a mapped solution (one chromosome of 100bags)
    ''' 
    The fitness function maximizes the value. 
    :param solution: a zipped solution made up of a randomly generated binary mask and the given bag population. 
    :return sum_weights: sum of the weights of the solution.
    :return sum_values: sum of the values of the solution.
    :return fitness: fitness of the solution. The fitness is the sum of the values.
    '''
    sum_weights = 0
    sum_values = 0
    s = []
    for bag in solution:
        if bag[0] == 1:
            sum_weights += list(bag[1].values())[0]['weight']
            sum_values += list(bag[1].values())[0]['value']
        else: pass
    fitness = sum_values 
    return [sum_weights, sum_values, fitness]

def generate_mask(pop_size):
    ''' This function generates a binary mask whose length is equivalent to the number of bags in a solution
        :param pop_size: population size
    '''

    mask = [np.random.choice([0,1]) for i in range(pop_size)]
    return mask

def tournament_selection(ms, t_size):
    ''' This function returns the fittest of two or more parents based on t_size
        :param t_size: tournament size
        :param ms: mapped solutions (100 solutions)
        Its arguments are a list containing a set of 100 solutions (1 solution=100bags) and t_size
    '''
    potential_parents = []; 
    fitness_potential_parents = []

    # Select n number of parents and get the fitness of each
    for i in range(t_size):
        potential_parents.append(random.choice(ms)) 
        fitness_potential_parents.append(fitness_function(potential_parents[i])[2])

    # choose the fittest    
    idx_highest_fitness = fitness_potential_parents.index(max(fitness_potential_parents))
    parent = potential_parents[idx_highest_fitness]
    
    return parent 

def crossover(p1,p2):
    ''' 
    Crossover the two selected parents at point, cp
    :param p1: parent 1
    :param p2: parent 2
    '''
    # let's get the binary mask of the selected parents 
    p1_mask = []
    for p in p1:
        p1_mask.append(p[0])
    
    p2_mask = []
    for i in p2:
        p2_mask.append(i[0])
    
    # randomly choose a crossover point (cp)
    cp = random.randrange(len(p1_mask))

    # crossover after point, cp
    new_p1 = p1_mask[cp+1:]
    new_p2 = p2_mask[cp+1:]
    
    child_1 = p1_mask.copy()
    child_1[cp+1:] = new_p2

    child_2 = p2_mask.copy()
    child_2[cp+1:] = new_p1

    return child_1, child_2 

def mutation(c,mut_rate):
    ''' 
    :param c: child to be mutated; 
    :param n: how many times the mutation is repeated in a child's chromosome
    '''
    # pos = the positions where mutation will take place
    pos = [random.randrange(100) for i in range(mut_rate)]
    for p in pos:
        if c[p] == 0: c[p] = 1
        else: c[p] = 0
    
    return c

def penalization_func(solution):
    '''
    This function penalizes a solution if its weight is > 285 (WEIGHT_LIMIT)
    It penalizes a solution by multiplying the fitness of the solution whose weight is above 285 by -1
    '''
    fitness_function_values = fitness_function(solution) # returns wght, values, fitness
    wght, val, fit = fitness_function_values
    
    # test for weights higher than the limit
    if wght > WEIGHT_LIMIT: 

        # multiply the fitness of the solution whose weight is above 285 by -1
        fit_adjusted = -1 * fit #- percent_wght_diff

        # the solution now has a lower fitness value
        fitness_function_values[2] = fit_adjusted
    else: pass
    
    return fitness_function_values

def replacement(sols, c1, c2):
    '''
    :param sols: solutions (the solutions, each with its mask)
    :param c1: child 1 (after crossover and mutation)
    :param c2: child 2 (after crossover and mutation)

    The function replaces a solution having lower fitness than any of the children.
    Replacement type used is first worst replacement.
    '''

    for i, v in enumerate(sols):
        # penalization_func returns 3 values: weight, value and fitness. 
        # Here we get the penalized fitness of any solution whose value is greater than 285
        pen_func_val = penalization_func(v)[2]

        # compare the penalized fitness with the fitness of the first child.
        # if child1's fitness is greater than that particular solution, remove the solution and replace with child
        if pen_func_val < fitness_function(c1)[2]:
            sols.remove(v) 
            sols.insert(i,c1)
            break

    # replace any other solution with child2 using the same condition as above
    for k,l in enumerate(sols[i+1:]):
        pen_func_val = penalization_func(l)[2]
        if pen_func_val < fitness_function(c2)[2]:
            sols.remove(l) 
            sols.insert(k,c2)
            break
    
    return sols

''' Bringing everything together '''
WEIGHT_LIMIT = 285

def evolutionary_algo(num_trials):
    
    for trial in range(1,num_trials+1):
        
        # generate n length number of solution masks (mask_len)
        binary_masks = []         # contains the masks
        for i in range(pop_size): 
            binary_masks.append(generate_mask(pop_size))

        # created a zipped list to hold the mask and population
        mapped_solutions = []
        for binary_mask in binary_masks:
            mapped_solution = list(zip(binary_mask, given_population))
            mapped_solutions.append(mapped_solution)
            

        # let's carry out 10,000 fitness evaluations
        count = 0
        while count < 10000:
            # Tournament Selection: Select any two parents from the mapped solutions
            parent_1 = tournament_selection(mapped_solutions,t_size)
            parent_2 = tournament_selection(mapped_solutions,t_size)

            ##### CROSSOVER
            child_c, child_d = crossover(parent_1,parent_2)

            ##### MUTATION
            child_e = mutation(child_c, mut_rate) # child e
            child_f = mutation(child_d, mut_rate) # child f 

            # Map the Children to the population 
            mapped_child_e = list(zip(child_e, given_population))
            mapped_child_f = list(zip(child_f, given_population))

            ##### Replacement
            # Replacement function evaluates the fitness and implements penalization of each solution
            # Penalization implemented in the replacement
            new_population = replacement(mapped_solutions, mapped_child_e, mapped_child_f)
            #len(new_s)

            chosen_sols = [] # keeps track of the solutions with weight <= 285kg
            chosen_sol_values = [] # keeps track of the weight, value and fitness of each solution with weight <= 285kg

            for ms in new_population:
                # if fitness_function(ms)[0] >= WEIGHT_LIMIT:
                #     pass
                # else:
                chosen_sols.append(ms)
                chosen_sol_values.append(fitness_function(ms)) 

                 
            best_val = []; best_wght = []; best_fit = []; highest_val_per_iter = []; highest_wght_per_iter = list()

            for i,cs in enumerate(new_population):
                best_wght.append(chosen_sol_values[i][0])
                best_val.append(chosen_sol_values[i][1])
                best_fit.append(chosen_sol_values[i][2])

            highest_val = max(best_val)
            idx = best_val.index(highest_val)


            wght = round(best_wght[idx],3)
            fit = round(best_fit[idx], 3)
            
            # solution with best fitness
            sol_best = chosen_sols[idx]

            print(f'Pop: {count}; value: {highest_val}; weight: {wght}; fitness: {fit} ')

            highest_val_per_iter.append(highest_val)
            highest_wght_per_iter.append(wght)
            

            best_val_after_10000_iter = max(highest_val_per_iter)
            best_wght_after_10000_iter = max(highest_wght_per_iter)
            
            count += 1

            mapped_solutions = new_population
            

    return best_val_after_10000_iter, best_wght_after_10000_iter

if __name__ == '__main__':

    # Define seed value
    random.seed(2345)
    
    # Define the population
    given_population = import_data()

    # Initialize variables
    pop_size = 100      # population size
    t_size = 10         # tournament size
    mut_rate = 2        # mutation rate
    
    # Run the algorithm
    best_values = evolutionary_algo(num_trials=1)
    print(best_values)
    
    print('Parameters: \n' 
        'Population size: {}; Tournament size: {}; Mutation Rate: {}'.format(pop_size,t_size,mut_rate) )
            
   
print('Time taken: {:.3f}secs'.format(time.time() - start_time))