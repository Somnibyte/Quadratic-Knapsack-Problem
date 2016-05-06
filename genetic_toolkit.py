import random
import linecache
import copy
from collections import namedtuple

# Help Functions
def find_item_in_cycle(obj,cyclelist):
     for cycles in cyclelist:
         for list_of_tuples in cycles:
             for tuples in list_of_tuples:
                 if obj == tuples[0]:
                     return True
                 else:
                     continue
     return False

# Class to represent biological processes
class BiologicalProcessManager:

    """
    Crossover Operators
    """
    @staticmethod
    def cycle_crossover(crossover_rate,parentOne, parentTwo):

        print("Working with {}".format(parentOne.genotype_representation))
        cycles = []  # format: cycles => [ cycle => [sub1=>[], sub2=>[]]]
        foudItem = False

        random_probability = random.random()

        if random_probability < crossover_rate:
            return (parentOne, parentTwo)
        else:
            for i, obj in enumerate(parentOne.genotype_representation):
                if find_item_in_cycle(obj, cycles) == True:
                    continue
                else:
                    # Hold the items of a cycle
                    subcycleone = []
                    subcycletwo = []
                    start = []
                    # Keep track of the cycle indexes
                    current_p1_index = i
                    current_p2_index = i
                    # Keep track of the beginning of the cycle so we can stop
                    start.append(parentOne.genotype_representation[current_p1_index])
                    # Use a flag to end the loop
                    cycle_finished = False
                    while(cycle_finished != True):
                        # store current_p1_index
                        subcycleone.append((parentOne.genotype_representation[current_p1_index],current_p1_index))
                        # Find current_p1_index in p2 and store in the second subcycle array
                        opposite_obj = parentTwo.genotype_representation[current_p1_index]
                        subcycletwo.append((opposite_obj,current_p1_index))
                        # FInd current_p2_index in parentOne and store it as the new index
                        current_p1_index = parentOne.genotype_representation.index(opposite_obj)

                        # Check if we have completed the cycle
                        if parentOne.genotype_representation[current_p1_index] in start:
                            cycle_finished = True
                        else:
                            continue
                    # Add the subcycles into the cycle list
                    cycles.append([subcycleone,subcycletwo])

            # Setup child genotypes
            print(cycles)
            child_one_genotype = [0 for x in range(0,len(parentOne.genotype_representation))]
            child_two_genotype = [0 for x in range(0,len(parentOne.genotype_representation))]
            for j, obj in enumerate(cycles):
                # Odd cycle pairs stay the same
                if (j+1) % 2 != 0:
                    print("WORKING WITH ODDS, J IS: {}".format(j+1))
                    print("OBJ1: {}".format(obj[0]))
                    print("OBJ2: {}".format(obj[1]))
                    for k, item in enumerate(obj[0]):
                        print("[ODD] PUTTING: {} IN CHILD ONE AT INDEX: {}".format(item[0],item[1]))
                        child_one_genotype[item[1]] = item[0]
                    for l, item in enumerate(obj[1]):
                        print("[ODD] PUTTING: {} IN CHILD TWO AT INDEX: {}".format(item[0],item[1]))
                        child_two_genotype[item[1]] = item[0]
                    print("CHILD ONE: {}, CHILD TWO: {}".format(child_one_genotype, child_two_genotype))
                if (j+1) % 2 == 0:
                    print("WORKING WITH EVENS, J IS: {}".format(j+1))
                    print("OBJ1: {}".format(obj[0]))
                    print("OBJ2: {}".format(obj[1]))
                    for k, item in enumerate(obj[0]):
                        print("[EVEN] PUTTING: {} IN CHILD ONE AT INDEX: {}".format(item[0],item[1]))
                        child_two_genotype[item[1]] = item[0]

                    for l, item in enumerate(obj[1]):
                        print("[EVEN] PUTTING: {} IN CHILD TWO AT INDEX: {}".format(item[0],item[1]))
                        child_one_genotype[item[1]] = item[0]
                    print("CHILD ONE: {}, CHILD TWO: {}".format(child_one_genotype, child_two_genotype))
            #print("CHILD ONE: {}".format(sorted(child_one_genotype)))
            #print("ChILD TWO: {}".format(sorted(child_two_genotype)))
            childOne = Chromosome(parentOne.numberOfObjectsReference)
            childOne.genotype_representation = child_one_genotype
            childOne.phenotype_representation = parentOne.phenotype_representation
            childTwo = Chromosome(parentOne.numberOfObjectsReference)
            childTwo.genotype_representation = child_two_genotype
            childTwo.phenotype_representation = parentOne.phenotype_representation
            return (childOne,childTwo)

    @staticmethod
    def order_crossover(crossover_rate, parentOne, parentTwo):
        #print("Working with {} and {} \n".format(parentOne.genotype_representation, parentTwo.genotype_representation))
        random_probability = random.random()

        if random_probability < crossover_rate:
            return (parentOne, parentTwo)
        else:
            # Create two crossover points
            pivotOne = random.randint(
                0, len(parentOne.genotype_representation) - 1)
            pivotTwo = random.randint(pivotOne, len(
                parentOne.genotype_representation) - 1)
            # Setup offspring
            child_genotype = [0 for i in range(
                0, len(parentOne.genotype_representation))]
            # Copy segment from P1 into child
            #print("STARTING SEGMENT P1")
            segmentRange = [x for x in range(pivotOne, pivotTwo + 1)]
            for i, gene in enumerate(parentOne.genotype_representation):
                if i in segmentRange:
                    child_genotype[i] = gene

            #print(child_genotype)
            # Copy segment from P2 into child
            #print("STARTING SEGMENT P2")

            for j, gene in enumerate(parentTwo.genotype_representation):
                for k, item in enumerate(child_genotype):
                    if item == 0 and gene not in child_genotype:
                        child_genotype[k] = gene
                    elif item == 0 and gene in child_genotype:
                        break

            for j, gene in enumerate(parentTwo.genotype_representation):
                for k, gene in enumerate(child_genotype):
                    #print("cur pos: {} with gene {}".format(j, gene))
                    if child_genotype[j] == 0 and gene not in child_genotype and j not in segmentRange:
                        child_genotype[j] = gene
                        #print("Placed {} in position {}".format(gene, j))
                    elif j in segmentRange and gene not in child_genotype :
                        for k, item in enumerate(child_genotype):
                            if child_genotype[k] == 0:
                                child_genotype[k] = gene
                                #print("[K] Placed {} in position {}".format(gene, k))
                                break
                    else:
                        continue
            print(child_genotype)
            # Create offspring
            child = Chromosome(parentOne.numberOfObjectsReference)
            child.genotype_representation = child_genotype
            # Generate the phenotype representation of the child
            child.phenotype_representation = parentOne.phenotype_representation
            #print("DONE!")
            return (child, 0)

    @staticmethod
    def pmx(crossover_rate, parentOne, parentTwo):
        random_probability = random.random()

        if random_probability < crossover_rate:
            return (parentOne, parentTwo)
        else:
            # Create two crossover points
            pivotOne = random.randint(
                0, len(parentOne.genotype_representation) - 1)
            pivotTwo = random.randint(pivotOne, len(
                parentOne.genotype_representation) - 1)
            #print("FIRST PIVOT POINT:{} ".format(pivotOne))
            #print("SECOND PIVOT POINT: {}".format(pivotTwo))
            # Setup offspring
            child_genotype = [0 for i in range(
                0, len(parentOne.genotype_representation))]
            # Copy segment from P1 into child
            segmentRange = [x for x in range(pivotOne, pivotTwo + 1)]
            #print("SEGMENT RANGE: {}".format(segmentRange))
            #print("FIRST INSTRUCTION")
            for i, gene in enumerate(parentOne.genotype_representation):
                if i in segmentRange:
                    #print("ADDED op:{} to index:{}".format(operation, i))
                    child_genotype[i] = gene
            # Copy segment from P2 into child
            #print("SECOND INSTRUCTION")
            for j, gene in enumerate(parentTwo.genotype_representation):
                if j in segmentRange:
                    # Check  P1
                    a_gene = parentOne.genotype_representation[j]
                    #print("OP in P1 SIDE: {}".format(op))
                    # Check where the element exists in P2
                    index_of_gene = parentTwo.genotype_representation.index(
                        a_gene)
                    #print("INDEX OF PREV OP IN P2: {}".format(index_of_op))
                    # Check if the operation already exists in the child
                    if gene not in child_genotype:
                        # Check if position is occupied in the child
                        if child_genotype[index_of_gene] == 0:
                            #print("ADDED op:{} to index:{}".format(operation, index_of_op))
                            child_genotype[index_of_gene] = gene
                            #print("CHILD NOW HAS OP: {} in index: {}".format(child_genotype[index_of_op],index_of_op ))

                        else:
                            while(True):

                                #print("INFINI LOOOOOOOP!")
                                # Check P1
                                a_gene = parentOne.genotype_representation[
                                    index_of_gene]
                                #print("[WHILE] OP in P1 SIDE: {}".format(a_gene))
                                # Check where the element exists in P2
                                index_of_gene = parentTwo.genotype_representation.index(
                                    a_gene)

                                if child_genotype[index_of_gene] == 0:
                                    #print("[WHILE] ADDED op:{} to index:{}".format(operation, index_of_op))
                                    child_genotype[index_of_gene] = gene
                                    #print("[WHILE] CHILD NOW HAS OP: {} in index: {}".format(child_genotype[index_of_op],index_of_op ))
                                    break
                                else:
                                    #print("[WHILE]  INDEX OP IS CURRENTLY: {}".format(index_of_op))
                                    continue

            # Copy the rest P2 into the child
            for k, gene in enumerate(parentTwo.genotype_representation):
                if k not in segmentRange:
                    if child_genotype[k] == 0:
                        #print("ADDED op:{} to index:{}".format(operation, k))
                        child_genotype[k] = gene

            # Create offspring
            child = Chromosome(parentOne.numberOfObjectsReference)
            child.genotype_representation = child_genotype
            # Generate the phenotype representation of the child
            child.phenotype_representation = parentOne.phenotype_representation
            # Return the new offspring
            #print("---------------------- \n")
            #print("FINISHED! RETURNING CHILD")
            #print("---------------------- \n")
            return (child, 0)

    """
    Mutation Operators
    """
    @staticmethod
    def swap_mutation(child):
        randindexone = random.randint(
            0, len(child.genotype_representation) - 1)
        randindextwo = random.randint(
            0, len(child.genotype_representation) - 1)
        if randindexone == randindextwo:
            while(True):
                randindextwo = random.randint(
                    0, len(child.genotype_representation) - 1)
                if randindextwo != randindexone:
                    break
                else:
                    continue
        temp = child.genotype_representation[randindexone]
        child.genotype_representation[
            randindexone] = child.genotype_representation[randindextwo]
        child.genotype_representation[randindextwo] = temp

    @staticmethod
    def insert_mutation(child):
        randindexone = random.randint(
            0, len(child.genotype_representation) - 1)
        randindextwo = random.randint(
            0, len(child.genotype_representation) - 1)
        if randindexone == randindextwo:
            while(True):
                randindextwo = random.randint(
                    0, len(child.genotype_representation) - 1)
                if randindextwo != randindexone:
                    break
                else:
                    continue
        saved_obj = child.genotype_representation[randindextwo]
        del child.genotype_representation[randindextwo]
        child.genotype_representation.insert(randindexone + 1, saved_obj)

    @staticmethod
    def scramble_mutation(child):
        randindexone = random.randint(
            0, len(child.genotype_representation) - 1)
        randindextwo = random.randint(
            0, len(child.genotype_representation) - 1)
        if randindexone == randindextwo:
            while(True):
                randindextwo = random.randint(
                    0, len(child.genotype_representation) - 1)
                if randindextwo != randindexone:
                    break
                else:
                    continue

        if randindexone > randindextwo:
            arr_subset = child.genotype_representation[
                randindexone:randindextwo]
            random.shuffle(arr_subset)
            child.genotype_representation[
                randindexone:randindextwo] = arr_subset
        else:
            arr_subset = child.genotype_representation[
                randindextwo:randindexone]
            random.shuffle(arr_subset)
            child.genotype_representation[
                randindextwo:randindexone] = arr_subset

    @staticmethod
    def inversion_mutation(child):
        randindexone = random.randint(
            0, len(child.genotype_representation) - 1)
        randindextwo = random.randint(
            0, len(child.genotype_representation) - 1)
        if randindexone == randindextwo:
            while(True):
                randindextwo = random.randint(
                    0, len(child.genotype_representation) - 1)
                if randindextwo != randindexone:
                    break
                else:
                    continue

        if randindexone > randindextwo:
            arr_subset = child.genotype_representation[
                randindexone:randindextwo]
            arr_subset.reverse()
            child.genotype_representation[
                randindexone:randindextwo] = arr_subset
        else:
            arr_subset = child.genotype_representation[
                randindextwo:randindexone]
            arr_subset.reverse()
            child.genotype_representation[
                randindextwo:randindexone] = arr_subset


# Class to represent chromosome
class Chromosome:

    fitness = None  # Chromosomes fitness
    phenotype_representation = None  # Phenotype representation

    def __init__(self, numOfObjects, genotype_representation=None):
        self.numberOfObjectsReference = numOfObjects
        self.knapsack_indicator = [1 for x in range(0, numOfObjects)]

        if genotype_representation == None:
            self.genotype_representation = []
            # Create a random permutation for the chromsome
            while(True):
                random_num = random.randint(1, self.numberOfObjectsReference)
                #print("Random num: {}".format(random_num))
                if random_num not in self.genotype_representation:
                    self.genotype_representation.append(random_num)
                else:
                    continue

                if len(self.genotype_representation) == self.numberOfObjectsReference:
                    break
        else:
            self.genotype_representation = genotype_representation

        self.length_of_encoding = len(self.genotype_representation)

        #print("DONE MAKING CHROMSOME")
    '''
	 Generates a fitness for all the chromosomes by aggregating their benefits/values
	'''

    def generateFitness(self):
        knapsack = copy.deepcopy(main_knapsack)
        fitnessScore = 0
        allTargetsReached = False
        target = 1
        list_of_permutations = []  # keeps track of the permutation
        indicator = copy.deepcopy(self.knapsack_indicator)

        while(True):
            for i, placement_of_object in enumerate(self.genotype_representation):
                # If you found the target then perform the algorithm
                if placement_of_object == target and placement_of_object not in list_of_permutations:
                    # Check if the knapsack is full
                    #print("CURRENT TARGET: {}".format(target))

                    if target > len(self.genotype_representation):
                        allTargetsReached = True
                        #print("TARGETS REACHED")
                        break
                    else:
                        #print("PROCESSING PHENO...")
                        # Identify if the object is a pair or a single object
                        phenotype = self.phenotype_representation[i]
                        if phenotype.type == "non-pair":
                            #print("{}: CURRENTLY WORKING WITH: {}".format(target,phenotype.representation))
                            # if the weight is greater, then mark it with a 0
                            # and add as 'checked'
                            if phenotype.representation.weight > knapsack.capacity:
                                list_of_permutations.append(
                                    placement_of_object)

                                indicator[i] = 0

                                # Check when to exit
                                if target == len(self.genotype_representation):
                                    allTargetsReached = True
                                    break
                                else:
                                    target += 1
                                continue
                            else:
                                list_of_permutations.append(
                                    placement_of_object)
                                # Subtract from the capacity so we know how
                                # much the knapsack was used
                                knapsack.capacity -= phenotype.representation.weight
                                fitnessScore += phenotype.representation.value

                                # Check when to exit
                                if target == len(self.genotype_representation):
                                    allTargetsReached = True
                                    break
                                else:
                                    target += 1
                                continue

                        elif phenotype.type == "pair":
                            #print("{}: CURRENTLY WORKING WITH: {}".format(target,phenotype.representation))
                            # if the weight is greater, then mark it with a 0
                            # and add as 'checked'
                            combined_weight = phenotype.representation[
                                0].weight + phenotype.representation[1].weight
                            if combined_weight > knapsack.capacity:
                                list_of_permutations.append(
                                    placement_of_object)
                                indicator[i] = 0

                                # Check when to exit
                                if target == len(self.genotype_representation):
                                    allTargetsReached = True
                                    break
                                else:
                                    target += 1

                                continue
                            else:
                                list_of_permutations.append(
                                    placement_of_object)
                                # Subtract from the capacity so we know how
                                # much the knapsack was used
                                knapsack.capacity -= combined_weight
                                fitnessScore = phenotype.representation[
                                    0].value + phenotype.representation[1].value + phenotype.pair_value

                                # Check when to exit
                                if target == len(self.genotype_representation):
                                    allTargetsReached = True
                                    break
                                else:
                                    target += 1

                                continue

                # otherwise continue to search for the target
                else:
                    continue

            if(allTargetsReached == True):
                #print("TARGET FLAG: {}".format(allTargetsReached))
                break

        # record the new fitness
        #print("ENDED EVAL.")
        self.knapsack_indicator = indicator
        self.fitness = fitnessScore


""" Create Global Knapsack Object """


class Knapsack:

    def __init__(self, capacity):
        self.capacity = capacity

# The knapsacks capacity will be edited later in the Population class
main_knapsack = Knapsack(0)


class Phenotype:
    representation = None
    pair_value = None

    def __init__(self, type, arr):
        self.type = type
        if type == "pair":
            phenotype_attributes = namedtuple(
                'phenotypeAttributes', 'value weight')
            firstObject = phenotype_attributes(arr[0], arr[1])
            secondObject = phenotype_attributes(arr[2], arr[3])
            self.representation = [firstObject, secondObject]
            self.pair_value = arr[4]
        elif type == "non-pair":
            # The object must be a normal object
            phenotype_attributes = namedtuple(
                'phenotypeAttributes', 'value weight')
            self.representation = phenotype_attributes(arr[0], arr[1])


class Population:

    population = []

    def __init__(self, size):
        self.population_size = size

    def select_parents(self, tournament):
        '''
                Tournament selection is being used to find two parents
        '''
        first_fittest_indiv = None
        second_fittest_indiv = None

        for individual in tournament:
            # Check if this indivudal is fitter than the current fittist
            # individual
            if first_fittest_indiv == None or individual.fitness > first_fittest_indiv.fitness:
                first_fittest_indiv = individual

        tournament.remove(first_fittest_indiv)

        for individual in tournament:
            # Check if this indivudal is fitter than the current fittist
            # individual
            if second_fittest_indiv == None or individual.fitness > second_fittest_indiv.fitness:
                second_fittest_indiv = individual

        # print("FIRST: {},  SECOND: {}".format(first_fittest_indiv.fitness,second_fittest_indiv.fitness))
        return (first_fittest_indiv, second_fittest_indiv)

    def initialize_population(self):
        '''
                Read from a file and create the chromosomes
        '''
        # Open data file
        dataFile = open('100object.txt', 'r')

        # Read the capacity of the knapsack
        capacityOfKnapsack = int(dataFile.read(4))

        # Set the newly found capacity to the capacity of the main knapsack
        main_knapsack.capacity = capacityOfKnapsack

        # Read how many objects there will be.
        numOfObjects = len(dataFile.readlines())

        # Create phenotype representation of chromosome
        phenotype_representation = []

        for i in range(0, numOfObjects):
            # Get a line from the file
            line = linecache.getline("100object.txt", 2 + i).split()
            # Convert the values in the string to integers
            phenotype_values = [int(x) for x in line]

            # Identify whether the object is a pair or a separate object
            if len(phenotype_values) == 2:
                    # Create phenotype object
                new_phenotype = Phenotype("non-pair", phenotype_values)
                phenotype_representation.append(new_phenotype)
            elif len(phenotype_values) == 5:
                # Create phenotype object
                new_phenotype = Phenotype("pair", phenotype_values)
                phenotype_representation.append(new_phenotype)

        # Create the initial population
        for j in range(0, self.population_size):
            # Create a new chromosomed
            new_chromosome = Chromosome(numOfObjects)
            #  Give each chromosome it's phenotype representation
            new_chromosome.phenotype_representation = phenotype_representation
            # Evaluate each chromosome
            new_chromosome.generateFitness()
            # Add the chromsome to the population
            self.population.append(new_chromosome)

        dataFile.close()
