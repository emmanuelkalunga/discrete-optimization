#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import time
from collections import namedtuple
from operator import attrgetter
from collections import namedtuple
from io import BytesIO, StringIO

Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])
Block = namedtuple("Block", ['value', 'room', 'estimate'])

class Node:
    
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

    # A Branch and Bound method
    value = 0
    weight = 0
    taken = [0]*len(items)
    optimum = 0

    # Get the optimal Estimate value

    # Method 1: Just add together the values of all items   
    # optimal_estimate = 0
    # K = capacity
    # for item in items:
    #     optimal_estimate += item.value
    #     K -= item.weight

    # Method 2: Use linear relaxation on constraint
    # Sort items my density
    item_dataframe = pd.read_csv(StringIO(input_data), sep=" ", header=None, names=['value', 'weight'])
    item_dataframe.drop([0], inplace=True)
    item_dataframe['density'] = item_dataframe.apply(lambda x: float(x.value)/x.weight, axis=1)
    item_dataframe.sort_values('density', ascending=False, inplace=True)
    item_dataframe.reset_index(drop=True, inplace=True)
    # print(item_dataframe)
    # Add 1 or 1/x to knapsack until Room left is 0
    optimal_estimate = 0
    K = capacity
    for index, item in item_dataframe.iterrows():
        if K <= 0:
            break
        ratio = max([1, item['weight']/K])
        optimal_estimate += item['value']/ratio
        K -= item['weight']/ratio 

    # print("Optimal room left: {}, optimal value: {}".format(K, optimal_estimate))
    
    # Build the bounded tree: Depth first
    t0 = time.time()
    root = Node(Block(0,capacity, optimal_estimate), items, is_root=True, max_depth=item_count)
    root.insert()
    print("Tree size: {}".format(size(root)))
    # root.print_tree()
    selected_tails = []
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
