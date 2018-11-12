#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import time
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i, int(parts[0]), int(parts[1])))

    # A dynamic programming using table intuition
    value = 0
    weight = 0
    taken = [0]*len(items)

    # The table intuition 
    #  Populating table
    t0 = time.time()
    solution_table = pd.DataFrame(0, index=range(capacity+1),columns= [0] + [x.index  for x in items])
    for item in items:
        print("Processing item {}".format(item.index))
        for cap in solution_table.index:
            if item.weight <= cap:
                # Add item if resulting value is higher
                result_value = item.value + ( (cap >= item.weight) and solution_table[item.index-1][cap-item.weight] ) or (0) 
                if result_value > solution_table[item.index-1][cap]:                    
                    solution_table[item.index][cap] = result_value
                else:
                    solution_table[item.index][cap] = solution_table[item.index-1][cap]
    
    # Trace back (get solution)
    K = capacity
    for item in reversed(items):
        if solution_table[item.index][K] > solution_table[item.index-1][K]:
            taken[item.index-1] = 1
            value += item.value
            weight += item.weight
            K = K - item.weight
    t1 = time.time()
    
    # print(solution_table)
    print("Taken: {}".format(taken))
    print("Total Weight: {}".format(weight))
    print("Total value: {}".format(value))
    print("Time: {}".format(t1-t0))

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
