#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import time
from collections import namedtuple
from operator import attrgetter
from collections import namedtuple
from io import BytesIO, StringIO
import numpy as np 

Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])
Block = namedtuple("Block", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken'])
Node = namedtuple("Node", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken'])

class Node_:
    
    def __init__(self, block, items, left=None, right=None, is_tail=False, is_root=False, is_valid=True, max_depth=0, depth=0, taken=None):
        """
        Initiate the node
        :param block: a namedtuple collection 
        """
        # Note: Inforce datatype of arguments
        self.block = block
        self.left = left
        self.right = right
        self.is_tail = is_tail
        self.is_root = is_root
        self.is_valid = is_valid
        self.max_depth = max_depth
        self.depth = depth
        self.items = items
        self.highest_value = 0
        if taken is None:
            self.taken = [1]*len(items)
        else:
            self.taken = taken
    
    def __set_highest_value__(self, value):
        self.highest_value = value

        
    def spawn_left_child(self):
        new_value = self.block.value + self.items[self.depth].value
        # print("Item {} in {}: {} => {}".format(self.depth, len(self.items),self.items[self.depth], new_value))
        new_room = self.block.room - self.items[self.depth].weight
        new_validity = new_room >= 0
        new_estimate = self.block.estimate
        new_depth = self.depth + 1  
        new_taken = (self.taken).copy()
        new_taken[self.depth] = 1
        # print("At level {}, Left child receives {}, and setting element {} to 1 => {}".format(self.depth, self.taken, self.depth, new_taken))

        highest_value = max( [self.highest_value, new_value] ) if new_validity else self.highest_value

        left_child = Node(block=Block(new_value,new_room,new_estimate), 
                            items=self.items, 
                            is_tail=(new_depth == self.max_depth) or (new_room < 0) or (new_value==new_estimate), 
                            is_valid= new_validity ,
                            max_depth=self.max_depth,
                            depth=new_depth,
                            taken = new_taken
                        )
        left_child.__set_highest_value__(highest_value)
        return left_child

    
    def spawn_right_child(self, highest_value):
        new_value = self.block.value
        new_room = self.block.room 
        new_depth = self.depth + 1
        # Get new optimistic estimate

        # Method 1
        # new_estimate = self.block.estimate - self.items[self.depth].value

        # Method 2
        # if new_depth == self.max_depth:
        #     new_estimate = new_value
        # else:
        #     item_table = pd.DataFrame(data=None, index=[x.index for x in self.items[self.depth+1:]], columns=['value', 'weight', 'density'])
        #     for item in self.items[self.depth+1:]:
        #         item_table.loc[item.index] = pd.Series({'value': item.value, 'weight': item.weight, 'density': item.value/item.weight})
        #     item_table.sort_values('density', ascending=False, inplace=True)
        #     item_table.reset_index(drop=True, inplace=True)
        #     # print(item_table)
        #     # Add 1 or 1/x to knapsack until Room left is 0
        #     optimal_estimate = new_value #The optimal should at least have the value at the current node
        #     K = new_room
        #     for index, item in item_table.iterrows():
        #         if K <= 0:
        #             break
        #         ratio = max([1, item['weight']/K])
        #         optimal_estimate += item['value']/ratio
        #         K -= item['weight']/ratio 
        #     new_estimate = optimal_estimate

        # Method 3: Something close to Method 1
        # if new_depth == self.max_depth:
        #     new_estimate = new_value
        # new_estimate = new_value + sum([x.value for x in self.items[new_depth:]]) 
        #    
        # Method 4: Optimize method 2  
        if new_depth == self.max_depth:
            new_estimate = new_value
        else:
            # Add to value at current depth (self.block.value) the obtimistic estimate (with linear relaxation) considering 
            # 1) item number depth is not taken, and 2) the available room in the knapsack  
            #  Sort remaining items per density
            remaining_items = sorted(self.items[self.depth+1:], key=attrgetter('density'), reverse=True)            
            optimal_estimate = new_value #The optimal should at least have the value at the current node
            K = new_room
            for item in remaining_items:
                if K <= 0:
                    break
                else:
                    if item.weight <= K:
                        optimal_estimate += item.value
                        K -= item.weight
                    else:
                        coef = K/item.weight
                        optimal_estimate += coef * item.value
                        K = 0 # There should not be any room left after a fraction of an item has been taken
            new_estimate = optimal_estimate               
        
        # print("==> Estimate. parent: {}, right-child: {} (child value: {}, child room: {})".format(self.block.estimate, new_estimate, new_value, new_room))

        new_taken = (self.taken).copy()
        new_taken[self.depth] = 0
        right_child = Node(block=Block(new_value,new_room,new_estimate), 
                            items=self.items, 
                            is_tail=(new_depth == self.max_depth) or (new_room < 0) or (new_value==new_estimate) or (new_estimate <= highest_value), # Note: also need to keep track of the highest feasible value found
                            is_valid=new_room >= 0,
                            max_depth=self.max_depth,
                            depth=new_depth,
                            taken=new_taken
                        )
        right_child.__set_highest_value__(highest_value) # Note: might not be needed. Updating this value in the left spawn might suffice
        return right_child


    def insert(self):
        """
        Insert a node in tree
        :param block: namedtuple
        """
        if not self.is_tail:
            # Insert left    
            self.left = self.spawn_left_child()
            # print("Left child => {}".format(self.left.taken))
            highest_value = self.left.highest_value
            if not self.left.is_tail:
                highest_value = self.left.insert()
                # self.highest_value = self.left.insert()
            # Insert right
            self.right = self.spawn_right_child(highest_value)
            # print("Right child => {}".format(self.right.taken))
            if not self.right.is_tail:
                highest_value = self.right.insert()
                # self.highest_value = highest_value
        return highest_value
    
    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print("[V:{}, R:{}, E:{}],   highest={},   taken:{}     {}".format(self.block.value, self.block.room, self.block.estimate, self.highest_value, self.taken, ("Valid tail!" if (self.is_tail and self.is_valid) else "")))
        if self.right:
            self.right.print_tree()


    def get_valid_tails(self, valid_tails=[]):
        if self.is_tail:
            if self.is_valid:
                valid_tails.append(self)
        else:  
            if self.left:
                self.left.get_valid_tails(valid_tails)
            if self.right:
                self.right.get_valid_tails(valid_tails)
         

    def children_count(self, root):
        """
        Returns the number of children
        """
        count = 0
        if self.left:
            count += 1
        if self.right:
            count += 1
        return count

# Computes the number of nodes in tree 
def size(node): 
    if node is None: 
        return 0 
    else: 
        return (size(node.left)+ 1 + size(node.right)) 

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
    
    highest_value = 0
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
    tails_ordered_list = list()  #Contains node that have been marked as tail and valid
    nodes_ordered_list = list() #Active nodes: should not contains leaves (any node that has been marked as tail)
    nodes_ordered_list.insert(root, 0)
    tails_ordered_list

    # def spawn_left_child(self):
    #     new_value = self.block.value + self.items[self.depth].value
    #     # print("Item {} in {}: {} => {}".format(self.depth, len(self.items),self.items[self.depth], new_value))
    #     new_room = self.block.room - self.items[self.depth].weight
    #     new_validity = new_room >= 0
    #     new_estimate = self.block.estimate
    #     new_depth = self.depth + 1  
    #     new_taken = (self.taken).copy()
    #     new_taken[self.depth] = 1
    #     # print("At level {}, Left child receives {}, and setting element {} to 1 => {}".format(self.depth, self.taken, self.depth, new_taken))

    #     highest_value = max( [self.highest_value, new_value] ) if new_validity else self.highest_value

    #     left_child = Node(block=Block(new_value,new_room,new_estimate), 
    #                         items=self.items, 
    #                         is_tail=(new_depth == self.max_depth) or (new_room < 0) or (new_value==new_estimate), 
    #                         is_valid= new_validity ,
    #                         max_depth=self.max_depth,
    #                         depth=new_depth,
    #                         taken = new_taken
    #                     )
    #     left_child.__set_highest_value__(highest_value)
    #     return left_child

    def split_node(node):
        left_value = node.value + items[node.depth].value
        left_room = node.room - items[node.depth].weight
        left_validity = left_room >= 0 and 
        left_estimate = node.estimate
        left_depth = node.depth + 1
        left_is_tail = (left_depth == max_depth) or (left_room < 0 ) or (left_value==left_estimate)
        left_taken = (node.taken).copy()
        left_taken[node.depth] = 1
        left_child = Node(left_value, left_room, left_estimate, left_depth, left_is_tail, left_validity, left_taken) # namedtuple("Node", ['value', 'room', 'estimate', 'depth', 'is_tail', 'is_valid', 'taken'])

        return (left_child, right_child)


    def branchout(nodes_list, tail_list, best_solution_node):
        if len(nodes_list) > 0:
            left_child, right_child = split_node(nodes_list[-1])
            if left_child.is_tail:
                # Add to tail list (not in active list), and remove parent from active list
                parent = nodes_list.pop(-1)
                if left_child.is_valid:
                    tail_list.append(left_child)
                    if left_child.value > best_solution_node.value:
                        best_solution_node = left_child
            else:
                # Add to active node list. Replacing parent
                nodes_list[-1] = left_child  # left child replaces parent

            if right_child.is_tail:
                # add to list of tails if the tail is valid
                if right_child.is_valid:
                    tail_list.append(right_child)
                    if right_child.value > best_solution_node.value:
                        best_solution_node = right_child
            else:
                # Only insert into list of active nodes if child is not a tail
                nodes_list.inserst(get_position_in_node_list(right_child, nodes_list), right_child)
            branchout(nodes_list, tail_list, best_solution_node)
            

    while(len(nodes_ordered_list)>0):
        right_child, left_child = branchout_node(nodes_ordered_list[-1])
    

    
    
    print("Tree size: {}".format(size(root)))
    # root.print_tree()
    selected_tails = list()
    
    root.get_valid_tails(valid_tails=selected_tails)
    print("There {} valid tails:".format(len(selected_tails)))
    # Print solution and select the best one
    best_solution_value = -1
    best_tail = None
    for  i,tail in enumerate(selected_tails):
        # print("Tail {} solution -> Value: {}, room left: {}, selected items: {}".format(i, tail.block.value, tail.block.room, tail.taken))
        if tail.block.value > best_solution_value:
            best_solution_value = tail.block.value
            best_tail = tail
    value = best_solution_value
    optimum = 0
    taken = best_tail.taken
    t1 = time.time()
    print("Algorithm duration: {}".format(t1-t0))
    print("solution:")

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
