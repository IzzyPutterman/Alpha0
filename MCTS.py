#Not sure where to evaluate the node in the neural network

N(s,a) = 'visit count'
W(s,a) = 'total action value'
Q(s,a) = 'mean action value'
P(s,a) = 'prior prob of selecting this action'

class MCTS:
  def __init__(self, nn):
    self.nn = nn
    self.Ns = {}
    self.Wsa = {}
    self.Qsa = {}
    self.Ps = {}
    
  def search(self, currBoard):
    if "game over":
      'done'
    if sNode not in 'evald':
      'eval with neural net'
      'initialize that branch'
      'add to seen'
    #Iterativeley select move with upper confidence bound until find a good node
    uMax, Move = -float_inf, -1 #incase nothing is found
    for a in 'possible moves':
      poss_u = Q + 2**(1/2) * P * (sum(N)**(1/2))/(1 + N(a)
      if poss_u > uMax:
         uMax = poss_u
         Move = a
    #or could do this
    Move = max(-1, max('possible moves', key=lambda a:  Q + 2**(1/2) * P * (sum(N)**(1/2))/(1 + N(a)))
    uMax = 'above evaluated'
    #Evaluate node in NN
    newGames = 'initGame updated for gMove'
    v = self.search(Move, nn, newGame)
    #Update visit count of nodes transversed
    N(sNode, a) += 1 
    #Update action value
    W(sNode, a) += v
    Q(sNode, a) = W(sNode, a)/ N(sNode, a)
    return v
