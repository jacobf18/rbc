# Reconnaissance Blind Chess (RBC)
This RBC bot is built utilizing the following main ides:
1. Deep Q Learning
2. Monte Carlo Tree Search (MCTS)
3. Self-play
4. Neural Episodic Control (NEC)
5. Causal Convolutions
6. (Novelty Search)

To start, the bot works in a very similar way to AlphaZero. To overcome the imperfect information part of the game, the previous N board configurations as part of the state. This allows the network to utilize temporal information about the opponent's pieces. Like AlphaZero, the bot is trained via self-play, which is stabilized by using MCTS. To increase the speed of training, NEC is added onto the network. To better search for potential moves in a manner similar to how humans search a space, a novelty metric is placed over the potential actions based on the encoding of the network.
