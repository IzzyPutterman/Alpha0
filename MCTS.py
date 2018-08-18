#Not sure where to evaluate the node in the neural network

N(s,a) = 'visit count'
W(s,a) = 'total action value'
Q(s,a) = 'mean action value'
P(s,a) = 'prior prob of selecting this action'

class MCTS:
  def __init__(self, nn, gameIns):
    self.nn = nn
    self.game = gameIns
    self.Ns = {}
    self.Nsa = {}
    self.Qsa = {}
    self.Ps = {}
    
  def search(self, currBoard):
    s = self.game.stringRepresentation(currBoard) #hashForm
    if self.game.getGameEnded(currBoard):
      return
    if s not in Ps:
      self.Ps[s], v = self.nn.predict()
      return -v
    #Iterativeley select move with upper confidence bound until find a good node
    uMax, Move = -float_inf, -1 #incase nothing is found
    for a in self.game.getValidMoves(currBoard):
      poss_u = Q[(s,a)] + 2**(1/2) * P[s] * (sum(N[s])**(1/2))/(1 + N[(s,a)]))
      if poss_u > uMax:
         uMax = poss_u
         Move = a
    #or could do this
    Move = max(-1, max('possible moves', key=lambda a:  Q + 2**(1/2) * P * (sum(N)**(1/2))/(1 + N(a)))
    uMax = 'above evaluated'
               
               
    newBoard, newPlayer = self.game.nextState(currBoard, 1, Move)
    v = self.search(newBoard)
    #Update visit count of nodes transversed
   
    #Update action value
    Q[(s, Move)] = (self.Nsa.get((s,Move),0) * self.Qsa.get((s,Move),0) + v)/(N[(s,Move)] + 1)
    N[(s,a)] += 1
    N[s] += 1 
    return -v
