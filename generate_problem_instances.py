# This file generates the problem instances for the Quadratic Knapsack Problem
import random

# Erase the file
f = open('main_data_file.txt', 'r+')
f.truncate()

filename = "main_data_file.txt"
capacity_of_knapsack = 100
data_file = open(filename, 'w')
data_file.write(str(capacity_of_knapsack))
data_file.write("\n")


for x in range(0,100):
	randomNum = random.randint(0,1)
	# If 0 then one object, if 1 then a pair of objects
	if randomNum == 0:
		random_weight = random.randint(0,capacity_of_knapsack-1)
		random_value = random.randint(0, capacity_of_knapsack-1)
		data_file.write(str(random_weight))
		data_file.write(' ')
		data_file.write(str(random_value))
		data_file.write(' ')
		data_file.write("\n")

	elif randomNum == 1:
		random_weight = random.randint(0,capacity_of_knapsack-1)
		random_value = random.randint(0, capacity_of_knapsack-1)
		random_weight_two = random.randint(0,capacity_of_knapsack-1)
		random_value_two = random.randint(0, capacity_of_knapsack-1)
		random_pair_value = random.randint(0, capacity_of_knapsack-1)
		data_file.write(str(random_weight))
		data_file.write(' ')
		data_file.write(str(random_value))
		data_file.write(' ')
		data_file.write(str(random_weight_two))
		data_file.write(' ')
		data_file.write(str(random_value_two))
		data_file.write(' ')
		data_file.write(str(random_pair_value))
		data_file.write(' ')
		data_file.write("\n")

print("DONE")
