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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # common design for evaluation functions is eval(s) = w_1*f_1(s) + ... + w_i*f_i(s)
        # where s = state, w_1 is some weight, f_i(s) = corresponds to a feature extracted from the input state s -> return numerical value
        # subtracting incentivizes towards the goals, and makes pac man move
        
        def compute_euclidean(pos_1, pos_2):
            x_1, y_1 = pos_1
            x_2, y_2 = pos_2
            return ((x_2 - x_1)**2 + (y_2 - y_1)**2)**0.5
        
        eval = 0
        if len(newFood.asList()) > 0:
            nearest = min([compute_euclidean(newPos, food) for food in newFood.asList()])
            eval -= nearest
       
        for ghost_state in newGhostStates:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = compute_euclidean(newPos, ghost_pos)
            if ghost_distance < 2: 
                eval -= 1000
        
        if sum(newScaredTimes) > 0:
            eval -= 500
                
        eval -= 10 * len(newFood.asList()) - successorGameState.getScore()
        return eval
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def max_value(state, depth):
            v = float('-inf')
            for act in state.getLegalActions(0):
                successor_state = state.generateSuccessor(0, act)
                v = max(v, value(successor_state, 1, depth))
            return v
        
        def min_value(state, curr_agent, depth):
            v = float('inf')
            for act in state.getLegalActions(curr_agent):
                successor_state = state.generateSuccessor(curr_agent, act)
                if curr_agent == gameState.getNumAgents() - 1:
                    v = min(v, value(successor_state, 0, depth + 1))
                else:
                    v = min(v, value(successor_state, curr_agent + 1, depth))
            return v
        
        def value(state, curr_agent, depth):
            if state.isLose() or state.isWin() or depth == self.depth:
                return self.evaluationFunction(state)
            if curr_agent == 0: 
                return max_value(state, depth)            
            else: 
                return min_value(state, curr_agent, depth)
        
        best_action = None
        best_value = float('-inf')

        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            value_of_action = value(successor_state, 1, 0)
            if value_of_action > best_value:
                best_value = value_of_action
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, alpha, beta):
            v = float('-inf')
            for act in state.getLegalActions(0):
                successor_state = state.generateSuccessor(0, act)
                v = max(v, value(successor_state, 1, depth, alpha, beta))
                alpha = max(alpha, v)
                if beta < alpha:
                    break
            return v
        
        def min_value(state, curr_agent, depth, alpha, beta):
            v = float('inf')
            for act in state.getLegalActions(curr_agent):
                successor_state = state.generateSuccessor(curr_agent, act)
                if curr_agent == gameState.getNumAgents() - 1:
                    v = min(v, value(successor_state, 0, depth + 1, alpha, beta))
                else:
                    v = min(v, value(successor_state, curr_agent + 1, depth, alpha, beta))
                beta = min(beta, v)
                if beta < alpha:
                    break
            return v
        
        def value(state, curr_agent, depth, alpha, beta):
            if state.isLose() or state.isWin() or depth == self.depth:
                return self.evaluationFunction(state)
            if curr_agent == 0: 
                return max_value(state, depth, alpha, beta)            
            else: 
                return min_value(state, curr_agent, depth, alpha, beta)
            
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')

        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            value_of_action = value(successor_state, 1, 0, alpha, float('inf'))
            if value_of_action > best_value:
                best_value = value_of_action
                best_action = action
                alpha = value_of_action

        return best_action
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
