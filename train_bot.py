from reconchess_custom.reconchess import *
from reconchess_custom.reconchess.bots.random_bot import RandomBot
from reconchess_custom.reconchess.bots.trout_bot import TroutBot

from bot import AlphaRBC

from Net import Net

def train(net, examples):
    pass

def self_play(num_iters = 1):
    net = Net()

    for i in range(num_iters):
        examples = []
        # Execute self-play game
        bot1 = AlphaRBC()
        bot2 = RandomBot()
        winner_color, win_reason, game_history = play.play_local_game(bot1, bot2)

        # examples.extend(bot1.examples)
        # examples.extend(bot2.examples)

        # train(net, examples)

    return net

def main():
    self_play()

if __name__ == "__main__":
    main()