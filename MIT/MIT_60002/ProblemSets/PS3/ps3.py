# -*- coding: utf-8 -*-
# Problem Set 3: Simulating robots
# Name:
# Collaborators (discussion):
# Time:

import math
import random

import ps3_visualize
import matplotlib.pyplot as plt

# For python 2.7:
from ps3_verify_movement27 import test_robot_movement


# === Provided class Position
class Position(object):
    """
    A Position represents a location in a two-dimensional room, where
    coordinates are given by floats (x, y).
    """
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y
        
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_new_position(self, angle, speed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does NOT test whether the returned position fits inside the room.

        angle: float representing angle in degrees, 0 <= angle < 360
        speed: positive float representing speed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.get_x(), self.get_y()
        
        # Compute the change in position
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))
        
        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        
        return Position(new_x, new_y)

    def __str__(self):  
        return "Position: " + str(math.floor(self.x)) + ", " + str(math.floor(self.y))


# === Problem 1
class RectangularRoom(object):
    """
    A RectangularRoom represents a rectangular region containing clean or dirty
    tiles.

    A room has a width and a height and contains (width * height) tiles. Each tile
    has some fixed amount of dirt. The tile is considered clean only when the amount
    of dirt on this tile is 0.
    """
    def __init__(self, width, height, dirt_amount):
        """
        Initializes a rectangular room with the specified width, height, and 
        dirt_amount on each tile.

        width: an integer > 0
        height: an integer > 0
        dirt_amount: an integer >= 0
        """

        self.width = int(width)
        self.height = int(height)
        self.dirt_amount = int(dirt_amount)

        self.tiles = {}
        for i in range(self.width):
            for j in range(self.height):
                self.tiles[(i, j)] = self.dirt_amount



    def clean_tile_at_position(self, pos: Position, capacity: int):
        """
        Mark the tile under the position pos as cleaned by capacity amount of dirt.

        Assumes that pos represents a valid position inside this room.

        pos: a Position object
        capacity: the amount of dirt to be cleaned in a single time-step
                  can be negative which would mean adding dirt to the tile

        Note: The amount of dirt on each tile should be NON-NEGATIVE.
              If the capacity exceeds the amount of dirt on the tile, mark it as 0.
        """

        pos_x = int( math.floor(pos.get_x()))
        pos_y = int( math.floor(pos.get_y()))
        tile = (pos_x, pos_y)

        assert(tile in self.tiles.keys())
        self.tiles[tile] -= capacity
        if self.tiles[tile] < 0:
            self.tiles[tile] = 0


    def is_tile_cleaned(self, m, n):
        """
        Return True if the tile (m, n) has been cleaned.

        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer
        
        Returns: True if the tile (m, n) is cleaned, False otherwise

        Note: The tile is considered clean only when the amount of dirt on this
              tile is 0.
        """

        return self.tiles[(m, n)] == 0

    def get_num_cleaned_tiles(self):
        """
        Returns: an integer; the total number of clean tiles in the room
        """
        clean_tiles = dict(filter(lambda elem: elem[1] == 0, self.tiles.items()))

        return len(clean_tiles)

    def is_position_in_room(self, pos: Position):
        """
        Determines if pos is inside the room.

        pos: a Position object.
        Returns: True if pos is in the room, False otherwise.
        """
        tile = (int(math.floor(pos.get_x())), int(math.floor(pos.get_y())) )
        return tile in self.tiles.keys()

    def get_dirt_amount(self, m, n):
        """
        Return the amount of dirt on the tile (m, n)
        
        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer

        Returns: an integer
        """

        return self.tiles[(m, n)]
        
    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        # do not change -- implement in subclasses.
        raise NotImplementedError 
        
    def is_position_valid(self, pos):
        """
        pos: a Position object.
        
        returns: True if pos is in the room and (in the case of FurnishedRoom) 
                 if position is unfurnished, False otherwise.
        """
        # do not change -- implement in subclasses
        raise NotImplementedError         

    def get_random_position(self):
        """
        Returns: a Position object; a random position inside the room
        """
        # do not change -- implement in subclasses
        raise NotImplementedError        


class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times, the robot has a particular position and direction in the room.
    The robot also has a fixed speed and a fixed cleaning capacity.

    Subclasses of Robot should provide movement strategies by implementing
    update_position_and_clean, which simulates a single time-step.
    """
    def __init__(self, room: RectangularRoom, speed: float, capacity: int):
        """
        Initializes a Robot with the given speed and given cleaning capacity in the 
        specified room. The robot initially has a random direction and a random 
        position in the room.

        room:  a RectangularRoom object.
        speed: a float (speed > 0)
        capacity: a positive integer; the amount of dirt cleaned by the robot
                  in a single time-step
        """
        self.room = room
        assert speed > 0.0
        self.speed = speed
        self.capacity = int (capacity)

        self.position = self.room.get_random_position()
        self.direction = random.random() * 360.0

    def get_robot_position(self):
        """
        Returns: a Position object giving the robot's position in the room.
        """
        return self.position

    def get_robot_direction(self):
        """
        Returns: a float d giving the direction of the robot as an angle in
        degrees, 0.0 <= d < 360.0.
        """
        return self.direction

    def set_robot_position(self, position: Position):
        """
        Set the position of the robot to position.

        position: a Position object.
        """
        self.position = position

    def set_robot_direction(self, direction: float):
        """
        Set the direction of the robot to direction.

        direction: float representing an angle in degrees
        """
        self.direction = direction

    def update_position_and_clean(self):
        """
        Simulate the raise passage of a single time-step.

        Move the robot to a new random position (if the new position is invalid, 
        rotate once to a random new direction, and stay stationary) and mark the tile it is on as having
        been cleaned by capacity amount. 
        """
        # do not change -- implement in subclasses
        raise NotImplementedError

# === Problem 2
class EmptyRoom(RectangularRoom):
    """
    An EmptyRoom represents a RectangularRoom with no furniture.
    """
    def __init__(self, width, height, dirt_amount):
        super().__init__(width, height, dirt_amount)

    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        return int(self.width * self.height)
        
    def is_position_valid(self, pos: Position):
        """
        pos: a Position object.
        
        Returns: True if pos is in the room, False otherwise.
        """

        x = int( math.floor(pos.get_x()))
        y = int( math.floor(pos.get_y()))

        return (x, y) in self.tiles.keys()

    def get_random_position(self):
        """
        Returns: a Position object; a valid random position (inside the room).
        """
        x = random.random() * float(self.width)
        y = random.random() * float(self.height)
        return Position(x,y)

class FurnishedRoom(RectangularRoom):
    """
    A FurnishedRoom represents a RectangularRoom with a rectangular piece of 
    furniture. The robot should not be able to land on these furniture tiles.
    """
    def __init__(self, width, height, dirt_amount):
        """ 
        Initializes a FurnishedRoom, a subclass of RectangularRoom. FurnishedRoom
        also has a list of tiles which are furnished (furniture_tiles).
        """
        # This __init__ method is implemented for you -- do not change.
        
        # Call the __init__ method for the parent class
        RectangularRoom.__init__(self, width, height, dirt_amount)
        # Adds the data structure to contain the list of furnished tiles
        self.furniture_tiles = []
        
    def add_furniture_to_room(self):
        """
        Add a rectangular piece of furniture to the room. Furnished tiles are stored 
        as (x, y) tuples in the list furniture_tiles 
        
        Furniture location and size is randomly selected. Width and height are selected
        so that the piece of furniture fits within the room and does not occupy the 
        entire room. Position is selected by randomly selecting the location of the 
        bottom left corner of the piece of furniture so that the entire piece of 
        furniture lies in the room.
        """
        # This addFurnitureToRoom method is implemented for you. Do not change it.
        furniture_width = random.randint(1, self.width - 1)
        furniture_height = random.randint(1, self.height - 1)

        # Randomly choose bottom left corner of the furniture item.    
        f_bottom_left_x = random.randint(0, self.width - furniture_width)
        f_bottom_left_y = random.randint(0, self.height - furniture_height)

        # Fill list with tuples of furniture tiles.
        for i in range(f_bottom_left_x, f_bottom_left_x + furniture_width):
            for j in range(f_bottom_left_y, f_bottom_left_y + furniture_height):
                self.furniture_tiles.append((i,j))             

    def is_tile_furnished(self, m, n):
        """
        Return True if tile (m, n) is furnished.
        """
        return (m, n) in self.furniture_tiles

    def is_position_furnished(self, pos: Position):
        """
        pos: a Position object.

        Returns True if pos is furnished and False otherwise
        """
        x = int(math.floor(pos.get_x()))
        y = int(math.floor(pos.get_y()))

        return self.is_tile_furnished(x, y)

    def is_position_valid(self, pos):
        """
        pos: a Position object.
        
        returns: True if pos is in the room and is unfurnished, False otherwise.
        """

        x = int( math.floor(pos.get_x()))
        y = int( math.floor(pos.get_y()))

        if not self.is_position_furnished(pos):
            return (x, y) in self.tiles.keys()
        else:
            return False

    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room that can be accessed.
        """
        return int(self.width * self.height) - int( len(self.furniture_tiles))
        
    def get_random_position(self):
        """
        Return`s: a Position object; a valid random position (inside the room and not in a furnished area).
        """
        tile_copy = self.tiles.copy()
        # get all non-furnished tiles
        for tile in self.furniture_tiles:
            if tile in tile_copy.keys():
                del tile_copy[tile]
        (x,y) = random.choice(list(tile_copy.keys()))
        return Position(x,y)

# === Problem 3
class StandardRobot(Robot):
    """
    A StandardRobot is a Robot with the standard movement strategy.

    At each time-step, a StandardRobot attempts to move in its current
    direction; when it would hit a wall or furtniture, it *instead*
    chooses a new direction randomly.
    """

    # def __init__(self):
    #     super().__init__()
    def update_position_and_clean(self):
        """
        Simulate the raise passage of a single time-step.

        Move the robot to a new random position (if the new position is invalid, 
        rotate once to a random new direction, and stay stationary) and clean the dirt on the tile
        by its given capacity. 
        """

        position = self.position.get_new_position(self.direction, self.speed)
        if not self.room.is_position_valid(position):
            self.set_robot_direction(random.random() * 360.0)
        else:
            self.position = position
            self.room.clean_tile_at_position(self.position, self.capacity)



# Uncomment this line to see your implementation of StandardRobot in action!
# test_robot_movement(StandardRobot, EmptyRoom)
# test_robot_movement(StandardRobot, FurnishedRoom)

# === Problem 4
class FaultyRobot(Robot):
    """
    A FaultyRobot is a robot that will not clean the tile it moves to and
    pick a new, random direction for itself with probability p rather
    than simply cleaning the tile it moves to.
    """
    p = 0.15

    @staticmethod
    def set_faulty_probability(prob):
        """
        Sets the probability of getting faulty equal to PROB.

        prob: a float (0 <= prob <= 1)
        """
        FaultyRobot.p = prob
    
    def gets_faulty(self):
        """
        Answers the question: Does this FaultyRobot get faulty at this timestep?
        A FaultyRobot gets faulty with probability p.

        returns: True if the FaultyRobot gets faulty, False otherwise.
        """
        return random.random() < FaultyRobot.p
    
    def update_position_and_clean(self):
        """
        Simulate the passage of a single time-step.

        Check if the robot gets faulty. If the robot gets faulty,
        do not clean the current tile and change its direction randomly.

        If the robot does not get faulty, the robot should behave like
        StandardRobot at this time-step (checking if it can move to a new position,
        move there if it can, pick a new direction and stay stationary if it can't)
        """
        if self.gets_faulty():
            self.set_robot_direction(random.random() * 360.0)
            return
        else:
            position = self.position.get_new_position(self.direction, self.speed)
            if not self.room.is_position_valid(position):
                self.set_robot_direction(random.random() * 360.0)
            else:
                self.position = position
                self.room.clean_tile_at_position(self.position, self.capacity)

        
    
# test_robot_movement(FaultyRobot, EmptyRoom)

# === Problem 5
def run_simulation(num_robots, speed, capacity, width, height, dirt_amount, min_coverage, num_trials,
                  robot_type):
    """
    Runs num_trials trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction min_coverage of the room.

    The simulation is run with num_robots robots of type robot_type, each       
    with the input speed and capacity in a room of dimensions width x height
    with the dirt dirt_amount on each tile.
    
    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    capacity: an int (capacity >0)
    width: an int (width > 0)
    height: an int (height > 0)
    dirt_amount: an int
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. StandardRobot or
                FaultyRobot)
    """
    trial_time = []
    for trial in range(num_trials):

        room = EmptyRoom(width, height, dirt_amount)
        robots = []
        for i in range(num_robots):
            robots.append(robot_type(room, speed, capacity))
        coverage = room.get_num_cleaned_tiles() / room.get_num_tiles()
        time_steps = 0
        # anim = ps3_visualize.RobotVisualization(num_robots, width, height, [])
        while coverage < min_coverage:
            time_steps += 1
            for robot in robots:
                robot.update_position_and_clean()
                # anim.update(room, robots)
                coverage = float(room.get_num_cleaned_tiles()) / room.get_num_tiles()
        # anim.done()

        trial_time.append(time_steps)

    mean_time = sum(trial_time)/float(len(trial_time))
    return mean_time

# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 5, 5, 3, 1.0, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.8, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.9, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 20, 20, 3, 0.5, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(3, 1.0, 1, 20, 20, 3, 0.5, 50, StandardRobot)))

# === Problem 6
#
# ANSWER THE FOLLOWING QUESTIONS:
#
# 1)How does the performance of the two robot types compare when cleaning 80%
#       of a 20x20 room?
#
#
# 2) How does the performance of the two robot types compare when two of each
#       robot cleans 80% of rooms with dimensions 
#       10x30, 20x15, 25x12, and 50x6?
#
#

def show_plot_compare_strategies(title, x_label, y_label):
    """
    Produces a plot comparing the two robot strategies in a 20x20 room with 80%
    minimum coverage.
    """
    num_robot_range = range(1, 11)
    times1 = []
    times2 = []
    for num_robots in num_robot_range:
        print ("Plotting", num_robots, "robots...")
        times1.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.25, 20, StandardRobot))
        times2.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.25, 20, FaultyRobot))
    plt.plot(num_robot_range, times1)
    plt.plot(num_robot_range, times2)
    plt.title(title)
    plt.legend(('StandardRobot', 'FaultyRobot'))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def show_plot_room_shape(title, x_label, y_label):
    """
    Produces a plot showing dependence of cleaning time on room shape.
    """
    aspect_ratios = []
    times1 = []
    times2 = []
    for width in [10, 20, 25, 50]:
        height = int (300/width)
        print ("Plotting cleaning time for a room of width:", width, "by height:", height)
        aspect_ratios.append(float(width) / height)
        times1.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, StandardRobot))
        times2.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, FaultyRobot))
    plt.plot(aspect_ratios, times1)
    plt.plot(aspect_ratios, times2)
    plt.title(title)
    plt.legend(('StandardRobot', 'FaultyRobot'))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# show_plot_compare_strategies('Time to clean 80% of a 20x20 room, for various numbers of robots','Number of robots','Time / steps')
# show_plot_room_shape('Time to clean 80% of a 300-tile room for various room shapes','Aspect Ratio', 'Time / steps')
