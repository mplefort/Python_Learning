###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """

    cow_dict = {}
    file = open(filename, "r")
    for line in file:
        cow_dict[line.split(",")[0]] =  int(line.split(",")[1][0])
    # print(cow_dict)

    file.close()
    print(filename, ": ", cow_dict)
    return cow_dict

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    trips = []
    trip = []
    remain_space = limit
    sorted_cows = sorted(cows.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_cows)
    for cow in sorted_cows:
        if cow[1] < remain_space:
            trip.append(cow[0])
            remain_space -= cow[1]
        else:
            trips.append(trip)
            trip = []
            remain_space = limit
            if cow[1] < remain_space:
                trip.append(cow[0])
                remain_space -= cow[1]
    if trip:
        trips.append(trip)
    print(trips)
    print("Trips #: {} -- trips: {}".format(len(trips), trips))

    return trips

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """

    for cow_partition in get_partitions(cows):
        weight_limit_breached = False
        for set in cow_partition:
            weight = 0
            for cow in set:
                weight += cows.get(cow)
            if weight > limit:
                weight_limit_breached = True

        if not weight_limit_breached:
            print("Trips #: {} -- trips: {}".format(len(cow_partition), cow_partition))
            return cow_partition


        
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    cows1 = load_cows('ps1_cow_data.txt')
    cows2 = load_cows('ps1_cow_data_2.txt')
    print()
    timer(greedy_cow_transport, cows1, 10)
    print()
    timer(greedy_cow_transport, cows2, 10)
    print()
    print()
    timer(brute_force_cow_transport, cows1, 10)
    print()
    timer(brute_force_cow_transport, cows2, 10)
    pass

def timer(func, cow_data, limit=10):
    start = time.time()
    func(cow_data, limit=10)
    end = time.time()
    print(end - start)


compare_cow_transport_algorithms()