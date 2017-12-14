This is largely a replication of the methods in AlphaGo Zero but with a little twist. Instead of just predicting game winners, we explicitly use a form of Deep Q Learning called Deep Double-Q Learning.
This method has been shown to increase generalization of the network. 


This project was written for CS221 at Stanford.
A note for future Stanford students: Please do not use any of the code in this repo without attribution, that is against the Honor Code.


Note that this project uses a modified version of the file 'go.py' contained in OpenAI Gym that is required for this code to function. Ie the self play portion requires that we disable the function of the go engine that play against agents by default in the Go environment.

References:
Silver, David, et al. "Mastering the game of go without human knowledge." Nature 550.7676 (2017): 354-359.\\
Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep Reinforcement Learning with Double Q-Learning." AAAI. 2016.
