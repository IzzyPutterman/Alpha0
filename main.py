from Trainer import Trainer
from Othello import OthelloGame
from nnet import ConvNet

if __name__ == "__main__":
    game = OthelloGame(6)
    nnet = ConvNet(game)
    trainer = Trainer(game, nnet)
    trainer.train()
