#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import time
from collections import namedtuple
from operator import attrgetter
from collections import namedtuple
from io import BytesIO, StringIO
import numpy as np 
import sys

# sys.setrecursionlimit(100000)

Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])
Block = namedtuple("Block", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken'])
Node = namedtuple("Node", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken'])

def get_position_in_node_list(node, mylist):
    step = max( [1, round(len(mylist)/4)] )
    i = np.random.randint(0, len(mylist))

    #get direction
    if node.estimate > mylist[i].estimate:
        direction = 1
    elif node.estimate < mylist[i].estimate:
        direction = -1
    else:
        return i  # if value is equal to mylist[i], inserst value at position i
    
    while (1):
        if direction > 0:
            i = min( [i+step, len(mylist)-1] )
            if node.estimate < mylist[i].estimate:
                if step == 1:
                    return i
                i -= step
                step = max( [1, round(step/2)] ) 
            elif node.estimate == mylist[i].estimate:
                return i
            elif node.estimate > mylist[i].estimate and i == (len(mylist)-1):
                return i + 1
        elif direction < 1:
            i = max( [i-step, 0] )
            if node.estimate > mylist[i].estimate:
                if step == 1:
                    return i + 1
                i += step
                step = max( [1, round(step/2)] ) 
            elif node.estimate == mylist[i].estimate:
                return i
            elif node.estimate < mylist[i].estimate and i == 0:
                return i

def get_position_in_list(value, mylist):
    step = max( [1, round(len(mylist)/4)] )
    i = np.random.randint(0, len(mylist))

    #get direction
    if value > mylist[i]:
        direction = 1
    elif value < mylist[i]:
        direction = -1
    else:
        return i  # if value is equal to mylist[i], inserst value at position i
    
    while (1):
        if direction > 0:
            i = min( [i+step, len(mylist)-1] )
            if value < mylist[i]:
                if step == 1:
                    return i
                i -= step
                step = max( [1, round(step/2)] ) 
            elif value == mylist[i]:
                return i
            elif value > mylist[i] and i == (len(mylist)-1):
                return i + 1
        elif direction < 1:
            i = max( [i-step, 0] )
            if value > mylist[i]:
                if step == 1:
                    return i + 1
                i += step
                step = max( [1, round(step/2)] ) 
            elif value == mylist[i]:
                return i
            elif value < mylist[i] and i == 0:
                return i
# for value in [0, 12, 13, 50, 51, 100, 200, 300, 600, 900, 999, 1000, 5000, 9999]:
#     t0 = time.time()
#     testlist = list(range(100000))
#     idx = get_position_in_list(value, testlist)
#     t1 = time.time()
#     print("getting {} took {} seconds ==> position: {}".format(value, t1-t0, idx))
  

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # define internal functions
    def split_node(node, highest_value):
        # print("||===> spliting node: [V:{}, R:{}, E:{}]".format(node.value, node.room, node.estimate))
        # Get left child
        left_value = node.value + items[node.depth].value
        left_room = node.room - items[node.depth].weight
        left_validity = left_room >= 0
        left_estimate = node.estimate
        left_depth = node.depth + 1
        left_is_tail = (left_depth == max_depth) or (left_room < 0 ) or (left_value==left_estimate)
        left_taken = (node.taken).copy()
        left_taken[node.depth] = 1
        left_child = Node(left_value, left_room, left_estimate, left_depth, left_is_tail, left_validity, left_taken) # namedtuple("Node", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken'])

        # Get right child
        right_value = node.value
        right_room = node.room
        right_depth = node.depth + 1
        #Get new optimistic estimate
        # Linear relaxation
        if right_depth == max_depth:
            right_estimate = right_value
        else:
            # Add to value at current depth (self.block.value) the obtimistic estimate (with linear relaxation) considering 
            # 1) item number depth is not taken, and 2) the available room in the knapsack  
            #  Sort remaining items per density
            remaining_items = sorted(items[node.depth+1:], key=attrgetter('density'), reverse=True)
            optimistic_estimate = right_value #The optimal should at least have the value at the current node
            K = right_room
            for item in remaining_items:
                if K <= 0:
                    break
                else:
                    if item.weight <= K:
                        optimistic_estimate += item.value
                        K -= item.weight
                    else:
                        coef = K/item.weight
                        optimistic_estimate += coef * item.value 
                        K = 0 # There should not be any room left after a fraction of an item has been taken
            right_estimate = optimistic_estimate
        right_is_tail=(right_depth == max_depth) or (right_room < 0) or (right_value==right_estimate) or (right_estimate <= highest_value) # Note: also need to keep track of the highest feasible value found
        right_taken = (node.taken).copy()
        right_taken[node.depth] = 0
        right_child = Node(right_value, right_room, right_estimate, right_depth, right_is_tail, right_room >= 0, right_taken)
        # print(": Left child: {}\n: Right child: {}".format(left_child, right_child))

        return (left_child, right_child)


    def branchout(nodes_list, tail_list, best_solution_node, tree_size):
        # print("====> len nodes_list: {}".format(len(nodes_list)))
        if len(nodes_list) > 0:
            if nodes_list[-1].estimate < best_solution_node.value:
                removed_node = nodes_list.pop(-1)
                tail_list, nodes_list, best_solution_node = branchout(nodes_list, tail_list, best_solution_node, tree_size)
            else:
                # print("/\/\/\-> Splitting node: {}".format(nodes_list[-1]))
                left_child, right_child = split_node(nodes_list[-1], best_solution_node.value)
                # print("/\/\/\-> L: {}, R: {}".format(left_child, right_child))
                tree_size["number_nodes"] += 2 
                if left_child.is_tail:
                    # Add to tail list (not in active list), and remove parent from active list
                    # print("==> In created left child: \n{}".format(left_child))
                    parent = nodes_list.pop(-1)
                    if left_child.is_valid:
                        tail_list.append(left_child)
                        if left_child.value > best_solution_node.value:
                            best_solution_node = left_child
                            # print("+ Left Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))
                else:
                    # Add to active node list. Replacing parent
                    nodes_list[-1] = left_child  # left child replaces parent
                    if left_child.value > best_solution_node.value:
                        best_solution_node = left_child
                        # print("+ Left Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))

                if right_child.is_tail:
                    # add to list of tails if the tail is valid
                    if right_child.is_valid:
                        tail_list.append(right_child)
                        if right_child.value > best_solution_node.value:
                            best_solution_node = right_child
                            # print("+ Right Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))
                else:
                    # Only insert into list of active nodes if child is not a tail
                    # print("### Processing right child: {}".format(right_child))
                    nodes_list.insert(get_position_in_node_list(right_child, nodes_list), right_child)
                    if right_child.value > best_solution_node.value:
                        best_solution_node = right_child
                        # print("+ Right Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))
                # print(":::: Recursion with best solution: {}".format(best_solution_node))
                tail_list, nodes_list, best_solution_node = branchout(nodes_list, tail_list, best_solution_node, tree_size)
        # print("::::::: Returned best solution: {}".format(best_solution_node))
        return (tail_list, nodes_list, best_solution_node)

    #######################################################################################################################################
    def branchout_iter(nodes_list, tail_list, best_solution_node, tree_size):
        # print("====> len nodes_list: {}".format(len(nodes_list)))
        if len(nodes_list) > 0:
            if nodes_list[-1].estimate < best_solution_node.value:
                removed_node = nodes_list.pop(-1)
                # tail_list, nodes_list, best_solution_node = branchout(nodes_list, tail_list, best_solution_node, tree_size)
            else:
                # print("/\/\/\-> Splitting node: {}".format(nodes_list[-1]))
                left_child, right_child = split_node(nodes_list[-1], best_solution_node.value)
                # print("/\/\/\-> L: {}, R: {}".format(left_child, right_child))
                tree_size["number_nodes"] += 2 
                if left_child.is_tail:
                    # Add to tail list (not in active list), and remove parent from active list
                    # print("==> In created left child: \n{}".format(left_child))
                    parent = nodes_list.pop(-1)
                    if left_child.is_valid:
                        tail_list.append(left_child)
                        if left_child.value > best_solution_node.value:
                            best_solution_node = left_child
                            # print("+ Left Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))
                else:
                    # Add to active node list. Replacing parent
                    nodes_list[-1] = left_child  # left child replaces parent
                    if left_child.value > best_solution_node.value:
                        best_solution_node = left_child
                        # print("+ Left Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))

                if right_child.is_tail:
                    # add to list of tails if the tail is valid
                    if right_child.is_valid:
                        tail_list.append(right_child)
                        if right_child.value > best_solution_node.value:
                            best_solution_node = right_child
                            # print("+ Right Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))
                else:
                    # Only insert into list of active nodes if child is not a tail
                    # print("### Processing right child: {}".format(right_child))
                    nodes_list.insert(get_position_in_node_list(right_child, nodes_list), right_child)
                    if right_child.value > best_solution_node.value:
                        best_solution_node = right_child
                        # print("+ Right Updating best_solution_node. New best solution value: {}".format(best_solution_node.value))
                # print(":::: Recursion with best solution: {}".format(best_solution_node))
                # tail_list, nodes_list, best_solution_node = branchout(nodes_list, tail_list, best_solution_node, tree_size)
        # print("::::::: Returned best solution: {}".format(best_solution_node))
        return (tail_list, nodes_list, best_solution_node)
        #######################################################################################################################################
  
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i, int(parts[0]), int(parts[1]), float(parts[0])/float(parts[1]) ))
    
    # highest_value = 0
    max_depth = item_count # Depth go from 0 (root) to item_count. with last item added or not added on the last level

    # A Branch and Bound method. Best first
    value = 0
    weight = 0
    taken = [0]*len(items)
    optimum = 0

    # Get the optimal Estimate value

    # Method 2: Use linear relaxation on constraint
    # Sort items my density
    item_dataframe = pd.read_csv(StringIO(input_data), sep=" ", header=None, names=['value', 'weight'])
    item_dataframe.drop([0], inplace=True)
    item_dataframe['density'] = item_dataframe.apply(lambda x: float(x.value)/x.weight, axis=1)
    item_dataframe.sort_values('density', ascending=False, inplace=True)
    item_dataframe.reset_index(drop=True, inplace=True)
    # print(item_dataframe)
    # Add 1 or 1/x to knapsack until Room left is 0
    optimistic_estimate = 0
    K = capacity
    for index, item in item_dataframe.iterrows():
        if K <= 0:
            break
        ratio = max([1, item['weight']/K])
        optimistic_estimate += item['value']/ratio
        K -= item['weight']/ratio

    # Build the bounded tree: Best first
    t0 = time.time()
    root = Node(0, capacity, optimistic_estimate, 0, False, True, [0]*item_count)  # namedtuple("Block", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken']) 
    best_solution_node = Node(-1, capacity, optimistic_estimate, 0, False, True, [0]*item_count)
    tails_ordered_list = list()  #Contains node that have been marked as tail and valid
    nodes_ordered_list = list() #Active nodes: should not contains leaves (any node that has been marked as tail)
    tree_size = {"number_nodes": 1}
    nodes_ordered_list.insert(0, root)
    while(len(nodes_ordered_list)>0):
        tails_ordered_list, nodes_ordered_list, best_solution_node = branchout_iter(nodes_list=nodes_ordered_list, tail_list=tails_ordered_list, best_solution_node=best_solution_node, tree_size=tree_size) # main branching line 
        
    # tails_ordered_list, nodes_ordered_list, best_solution_node = branchout(nodes_list=nodes_ordered_list, tail_list=tails_ordered_list, best_solution_node=best_solution_node, tree_size=tree_size) # main branching line 
    # print(len(tails_ordered_list))
    # for node in tails_ordered_list:
    #     print(node)
    print("Tree size: {}".format(tree_size["number_nodes"]))
    value = best_solution_node.value
    optimum = 0
    taken = best_solution_node.taken
    t1 = time.time()
    print("Duration: {}".format(t1-t0))
    print("solution:")  
    # print(best_solution_node)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(optimum) + '\n'
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
