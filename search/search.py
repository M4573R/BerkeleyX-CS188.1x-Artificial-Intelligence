# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def graphSearch(problem, fringe):
    visited = set()
    path = [(problem.getStartState(), "", 0)]
    fringe.push(path)

    while not fringe.isEmpty():
        currentPath = fringe.pop()
        currentState, _action, _cost = currentPath[-1]
        if currentState in visited:
            continue

        if(problem.isGoalState(currentState)):
            return currentPath
        else:
            for (successor, cost, action) in problem.getSuccessors(currentState):
                fringe.push(currentPath + [(successor, cost, action)])
            visited.add(currentState)

def getListOfActions(path):
    actions = []
    for (state, action, cost) in path:
        if action is not "":
            actions.append(action)
    return actions

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    path = graphSearch(problem, fringe)
    return getListOfActions(path)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    path = graphSearch(problem, fringe)
    return getListOfActions(path)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
    
class SearchState:
    def __init__(self, coordinates, plan, accumulative_cost):
        self.coordinates = coordinates
        self.plan = plan
        self.accumulative_cost = accumulative_cost

    def add(self, coordinates, action, cost):
        return SearchState(coordinates, self.plan + [action], self.accumulative_cost + cost)

def prioritySearch(problem, heuristic=nullHeuristic):
    state = problem.getStartState()
    fringe = util.PriorityQueue()
    plan = []
    visited = set()
    currentState = SearchState(problem.getStartState(), [], 0)
    fringe.push(currentState, currentState.accumulative_cost)

    while(True):
       currentState = fringe.pop()
       if currentState.coordinates in visited:
           continue
       if(problem.isGoalState(currentState.coordinates)):
           return currentState.plan
       else:
           for (successor, action, cost) in problem.getSuccessors(currentState.coordinates):
               newState = currentState.add(successor, action, cost)
               fringe.push(newState, newState.accumulative_cost + heuristic(newState.coordinates, problem))
           visited.add(currentState.coordinates)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return prioritySearch(problem)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return prioritySearch(problem, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
