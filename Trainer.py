from MCTS import MCTS
import numpy as np

training_params = {
    'num_iterations': 100,
    'num_episodes': 100,
    'update_threshold': 0.55,
    'max_training_examples_per_iteration': 100000,
    'num_of_validation_games': 50,
    'max_complete_examples': 50,
}

class Trainer:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.pnet = nnet.__class__(self.game)
        self.args = training_params
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.complete_training_history = []

    def executeEpisode(self):
        training_data =  []
        board = self.game.getInitBoard()
        current_player = 1
        episode_number = 0

        while True:
            episode_number += 1

            # use the MCTS to get the move probabilities
            canonical_board = self.game.getCanonicalForm(board, current_player)
            pi = self.mcts.getProbabilityDist(canonical_board)
            sym = self.game.getSymmetries(canonical_board, pi)
            for sym_board, sym_pi in sym:
                training_data.append([sym_board, current_player, sym_pi])

            # choose a random move based on pi
            move = np.random.choice(len(pi), p=pi)
            board, current_player = self.game.getNextState(board, current_player, move)

            # see if the game is over which would be the reward
            r = self.game.getGameEnded(board, current_player)

            if r != 0:
                return [(x[0], x[2], r * ((-1)**(current_player!=x[1]))) for x in training_data]

    def train(self):
        print("Starting the training")
        for i in range(self.args['num_iterations']):
            print("Training outer iteration: " + str(i))
            training_examples = []
            for j in range(self.args['num_episodes']):
                print("Training inner iteration: " + str(j))
                self.mcts = MCTS(self.game, self.nnet, self.args)
                training_examples.append(self.executeEpisode())
                if (len(training_examples) > self.args['max_training_examples_per_iteration']):
                    training_examples.pop_front()

            self.complete_training_history.append(training_examples)
            if(len(self.complete_training_history) > self.args['max_complete_examples']):
                self.complete_training_history.pop_front()

            self.nnet.save("nnetsave")
            self.pnet.load("nnetsave")

            training_examples = []
            for example_sequence in self.complete_training_history:
                training_examples.extend(example_sequence)
            shuffle(training_examples)
            print("Training nnet")
            self.nnet.train(training_examples)

            wins, draws, losses = validate(self.nnet, self.pnet)
            print("Wins " + str(wins))
            print("Losses " + str(losses))
            print("Draws " + str(draws))
            # only update if win percentage is greater than 55
            if float(wins)/(float(wins+losses)) < self.args['update_threshold'] and wins + losses > 0:
                self.nnet = self.nnet.load("nnetsave")
            else:
                self.nnet.save("nnetbest")


    def validate(net1, net2):
        '''
        Determine win rate of net1
        '''
        # Make fresh MCTS for each nnet
        net1MCTS = MCTS(self.game, self.net1, self.args)
        net2MCTS = MCTS(self.game, self.net2, self.args)

        # Functions for greedy moves
        playerFunctions = {
            1: lambda x: np.argmax(net1MCTS.getProbabilityDist(x, temp = 0)),
            -1: lambda x: np.argmax(net2MCTS.getProbabilityDist(x, temp = 0))
        }

        # To store win rates 0 -> losses
        wins = {-1:0, 0:0, 1:0}
        
        current_starting_player = 1
        current_opposing_player = -1
        current_player = current_starting_player
        for i in range(self.args['num_of_validation_games']):
            current_player = current_starting_player
            current_board = self.game.getInitBoard()
            while self.game.getGameEnded(current_board, current_player) == False:
                action = playerFunction[current_player](self.game.getCanonicalForm(current_board, current_player))
                if self.game.getValidMoves(self.game.getCanonicalForm(current_board, current_player), 1)[action]==0:
                    break
                current_board, current_player = self.game.getNextState(current_board, current_player, action)
            wins[self.game.getGameEnded(current_board, 1)] += 1
            current_starting_player, current_opposing_player = current_opposing_player, current_starting_player
        return wins[1], wins[0], wins[-1]
