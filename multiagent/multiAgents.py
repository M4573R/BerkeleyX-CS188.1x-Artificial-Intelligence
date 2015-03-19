# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from operator import add

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        distToClosestFood = -1
        for food in newFood.asList():
            dist = manhattanDistance(newPos, food)
            if distToClosestFood < 0 or dist < distToClosestFood:
                distToClosestFood = dist

        distToClosestGhost = -1
        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            if distToClosestGhost < 0 or dist < distToClosestGhost:
                distToClosestGhost = dist

        foodEaten = len(currentGameState.getFood().asList()) - len(newFood.asList())
        if distToClosestGhost < 3:
            return 1.0 / distToClosestFood + distToClosestGhost + foodEaten
        else:
            return 10.0 / distToClosestFood + 100 * foodEaten

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        val = float('-inf')
        nextAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextVal = self.minValue(0, 1, gameState.generateSuccessor(0, action))
            if nextVal > val and action != Directions.STOP:
                val = nextVal
                nextAction = action

        return nextAction

    def maxValue(self, depth, agent, state):
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if len(actions) > 0:
                v = float('-inf')
            else:
                v = self.evaluationFunction(state)

            for action in actions:
                s = self.minValue(depth, agent+1, state.generateSuccessor(agent, action))
                if s > v:
                    v = s
            return v

    def minValue(self, depth, agent, state):
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
			actions = state.getLegalActions(agent)
			if len(actions) > 0:
				v = float('inf')
			else:
				v = self.evaluationFunction(state)

			for action in actions:
				if agent == state.getNumAgents() - 1:
					s = self.maxValue(depth+1, 0, state.generateSuccessor(agent, action))
					if s < v:
						v = s
				else:
					s = self.minValue(depth, agent+1, state.generateSuccessor(agent, action))
					if s < v:
						v = s
			return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alphas = [float('-inf')]
        for i in range(1, gameState.getNumAgents()):
            alphas.append(float('inf'))

        def getNextValue(state, depth, alphas):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth*state.getNumAgents():
                return self.evaluationFunction(state)
            if depth % state.getNumAgents() == 0:
                return maxValue(state, depth, depth % state.getNumAgents(), alphas)
            else:
                return minValue(state, depth, depth % state.getNumAgents(), alphas)

        def maxValue(state, depth, agentIndex, alphas):
            alphas = alphas[:]
            maxVal = float('-inf')
            for action in state.getLegalActions(agentIndex):
                maxVal = max(maxVal, getNextValue(state.generateSuccessor(agentIndex, action), depth+1, alphas))
                if maxVal > min(alphas[1:]):
                    return maxVal
                alphas[agentIndex] = max(alphas[agentIndex], maxVal)
            return maxVal

        def minValue(state, depth, agentIndex, alphas):
            alphas = alphas[:]
            minVal = float('inf')
            for action in state.getLegalActions(agentIndex):
                minVal = min(minVal, getNextValue(state.generateSuccessor(agentIndex, action), depth+1, alphas))
                if minVal < alphas[0]:
                    return minVal
                alphas[agentIndex] = min(alphas[agentIndex], minVal)
            return minVal

        val = float('-inf')
        nextAction = 0
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(0):
            nextVal = getNextValue(gameState.generateSuccessor(0, action), 1, alphas)
            if nextVal > val:
                nextAction = action
                val = nextVal
            alphas[0] = max(alphas[0], val)
        return nextAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def getNextValue(state, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if depth == self.depth*state.getNumAgents():
                return self.evaluationFunction(state)
            if depth % state.getNumAgents() == 0:
                return maxValue(state, depth, depth % state.getNumAgents())
            else:
                return minValue(state, depth, depth % state.getNumAgents())

        def maxValue(state, depth, agentIndex):
            actions = state.getLegalActions(agentIndex)
            return reduce(max, map(lambda x: getNextValue(state.generateSuccessor(agentIndex, x), depth+1), actions))

        def minValue(state, depth, agentIndex):
            actions = state.getLegalActions(agentIndex)
            return reduce(add, map(lambda x: getNextValue(state.generateSuccessor(agentIndex, x), depth+1), actions)) / len(actions)

        val = float('-inf')
        nextAction = 0
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(0):
            nextVal = getNextValue(gameState.generateSuccessor(0, action), 1)
            if nextVal > val:
                nextAction = action
                val = nextVal
        return nextAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: current game score with penaties for food left, distance to 
                   food and distance to ghosts
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    distToFood = 0
    for food in newFood.asList():
        distToFood += manhattanDistance(newPos, food)

    distToGhosts = 0
    for ghost in newGhostStates:
        distToGhosts += manhattanDistance(newPos, ghost.getPosition())

    foodEaten = len(currentGameState.getFood().asList()) - len(newFood.asList())

    val = 0
    val += 1.0 / (1 + len(newFood.asList()))
    val += 1.0 / (1 + distToFood)
    val += 1.0 / (1 + distToGhosts)
    val += currentGameState.getScore()

    return val

# Abbreviation
better = betterEvaluationFunction
