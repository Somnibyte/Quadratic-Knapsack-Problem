# Author: Guled
# Problem: Quadratic Knapsack Problem
from genetic_toolkit import Population,Chromosome,BiologicalProcessManager
import statistics
import random

'''
	Generation Evaluation Function
'''
# Data files

filename = "cycle/cycle_scramble100[1000gens].txt"
data_file = open(filename, 'w')



# Global Variables
crossover_rate = 0.70

# Initialize population with random candidate solutions
population = Population(100)
population.initialize_population()
# Get a reference to the number of knapsacks
#numberOfKnapsacks = population.numberOfKnapsacks
indiv = population.population[0]

# Main Algorithm
generation_counter = 0
test_counter = 0
print("Working..")
while(test_counter != 30):
	print("Test_counter {}".format(test_counter))
	while(generation_counter != 1000):
		current_population_fitnesses = [chromosome.fitness for chromosome in population.population]
		print("CURRENT GEN FITNESS: {} ".format(current_population_fitnesses))
		new_gen = []
		while(len(new_gen) <= population.population_size):
			# Create tournament for tournament selection process
			tournament = [population.population[random.randint(1, population.population_size-1)] for individual in range(1, population.population_size)]
			#print("LENGTH OF TOURN: {}".format(len(tournament)))
			# Obtain two parents from the process of tournament selection
			parent_one, parent_two = population.select_parents(tournament)

			# Create the offspring from those two parents
			child_one,child_two = BiologicalProcessManager.cycle_crossover(crossover_rate,parent_one,parent_two)

			# Our algorithm produces only one child or returns back to parents
			# Here we check if only one child was returned or not

				# Mutate the child
			BiologicalProcessManager.scramble_mutation(child_one)
			BiologicalProcessManager.scramble_mutation(child_two)

				# Evaluate the child
			child_one.generateFitness()
			child_two.generateFitness()

			new_gen.append(child_one)
			new_gen.append(child_two)

		# Replace old generation with the new one
		population.population = new_gen
		generation_counter += 1

	# Increment the test counter
	test_counter+=1
	# Pick out the largest fitness from the current population and write it to the file
	current_fitnesses = [chromosome.fitness for chromosome in population.population]
	data_file.write("Test: {} \n".format(test_counter))
	data_file.write("Result: {}".format(str(max(current_fitnesses))))
	data_file.write("\n")





print("\n")
for i, gene in enumerate(population.population[0].genotype_representation):
	print("{}: {}".format(i,gene))

print("\n")
for i, indicator in enumerate(population.population[0].knapsack_indicator):
	print("{}: {}".format(i,indicator))

print("\n")
for i, pheno in enumerate(population.population[0].phenotype_representation):
	print("{}: {}".format(i,pheno.representation))
