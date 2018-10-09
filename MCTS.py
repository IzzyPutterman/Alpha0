import numpy as np
import math

MCTS_params = {
    'num_sims': 50,
    'cpuct': 1,
}

class MCTS:
  def __init__(self, gameIns, nn, args):
    self.nn = nn
    self.args = MCTS_params
    self.game = gameIns
    self.N = {}  # Number of hits for a state (sum over NAction)
    self.NAction = {} # Number of hits for a state action pair
    self.Q = {}  # State action values for a state action pair
    self.P = {}  # Probabilities from nnet for a state (memoized)

  def getProbabilityDist(self, board, temp = 1):
      # execute simulations to expand the tree
      for i in range(self.args['num_sims']):
          self.search(board)

      # hash form to extract results
      s = self.game.stringRepresentation(board)
      counts = []

      # Get result distribution after search
      for a in range(self.game.getActionSize()):
          if (s,a) in self.NAction:
              counts.append((self.NAction[(s,a)])**(1/temp))
          else:
              counts.append(0)

      # Return after normalization
      return [i/float(sum(counts)) for i in counts]


  def search(self, currBoard):
    # This is the hash representation of the board
    s = self.game.stringRepresentation(currBoard) #hashForm

    # need to return a reward for how the game ended.
    if self.game.getGameEnded(currBoard, 1):
      return -self.game.getGameEnded(currBoard, 1)

    # Memoizing the masked probability distributions
    if s not in self.P:
      self.P[s], v = self.nn.predict(currBoard)
      self.P[s] = self.P[s] * self.game.getValidMoves(currBoard, 1)
      total = np.sum(self.P[s])
      print("total " + str(total))
      if total > 0:
          self.P[s] /= total
      else:
          print("Probability normalization factor is 0")
          self.P[s] = self.P[s] + self.game.getValidMoves(currBoard, 1)
          total = np.sum(self.P[s])
          if total > 0:
              self.P[s] /= total
      self.N[s] = 0
      return -v

    # Iterativeley select move with upper confidence bound until find a leaf
    u_max, move_max = -float('inf'), -1
    for a in range(self.game.getActionSize()):
        if self.game.getValidMoves(currBoard, 1)[a]:
            if (s,a) in self.Q:
                u_current = self.Q[(s,a)] + self.args['cpuct']*self.P[s][a]*math.sqrt(self.N[s])/(1+self.NAction[(s,a)])
            else:
                u_current = 0
            if u_current > u_max:
                u_max = u_current
                move_max = a

    # get the state of the next iteration
    new_board, new_player = self.game.getNextState(currBoard, 1, move_max)
    new_board = self.game.getCanonicalForm(new_board, new_player)
    v = self.search(new_board)

    # need to check whether the tuple exists before settings it.
    if (s,move_max) in self.Q:
        self.Q[(s,move_max)] = (self.NAction[(s, move_max)] * self.Q[(s,move_max)] + v)/(self.NAction[(s, move_max)] + 1)
        self.NAction[(s, move_max)] += 1
    else:
        self.Q[(s, move_max)] = v
        self.NAction[(s, move_max)] = 1

    # Guaranteed to be set by first loop
    self.N[s] += 1

    return -v
