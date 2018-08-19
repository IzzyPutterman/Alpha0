#Not sure where to evaluate the node in the neural network

class MCTS:
  def __init__(self, nn, gameIns):
    self.nn = nn
    self.game = gameIns
    self.N = {}
    self.Q = {}
    self.P = {}
    
  def search(self, currBoard):
    s = self.game.stringRepresentation(currBoard) #hashForm
    if self.game.getGameEnded(currBoard):
        # need to return a reward for how the game ended.
      return -self.game.getGameEnded(currBoard, 1)
    if s not in self.P:
      self.P[s], v = self.nn.predict()
      return -v
    #Iterativeley select move with upper confidence bound until find a leaf
    uMax, move = -float_inf, -1 #incase nothing is found
    for a in self.game.getValidMoves(currBoard, 1):
      poss_u = self.Q[s][a] + 2**(1/2) * self.P[s] * (sum(N[s])**(1/2))/(1 + N[s][a]))
      if poss_u > uMax:
         uMax = poss_u
         move = a
  
    newBoard, newPlayer = self.game.getNextState(currBoard, 1, move)
    newBoard = self.game.getCanonicalForm(newBoard, newPlayer)
    v = self.search(newBoard)
    #Update visit count of nodes transversed
   
    #Update action value
    # need to check whether the tuple exists before settings it.
    Q[s][move] = (self.N[s][move] * self.Q[s][move] + v)/(N[s][move] + 1)
    if s in self.N:
        if a in self.N[s]:
            self.N[s][a] +=1
    else:
        self.N[s][a] = 1

    return -v
