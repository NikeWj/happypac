# Pacman AI Methods

![Python Version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)

Astart Search vs baselineTeam
```
python3 capture.py -r astart.py -b baselineTeam.py -l RANDOM
```

Q-Approximate Learning vs baselineTeam
```
python3 capture.py -r QAL.py -b baselineTeam.py -l RANDOM
```

AttackAgent vs DefenceAgent (For training)
```
python3 capture.py -r attackAgent.py -b DefenceAgent.py -l RANDOM
```

```
Final Version is myTeam
```


<!-- ---------------------------------------------------------- -->

# Project Analysis and Design Decisions
By analyzing the rules, we summarize these points, which may affect our choice of technology:

* **Multi-agent and multi-function**: This is a multi-agent game, we have two agents. In these two agents, you can choose to attack (go to the opponent's area to eat beans) or defend (defend in our area). In order to achieve the highest efficiency, we need a good offensive and defensive conversion model. On offense, the two attack together, and if they encounter beans, they will double-team and defend. Because it is a multi-agent game, we can use game theory to choose the best choice for us.

* **Generalization**: The map of the game is randomly selected, for which we cannot use a specific method for a certain map. We don’t know the exact route of the enemy, we need good generalization ability.
 
 * **Time Limit**: There is 15 seconds of preparation time when the game starts. The general method needs to be placed in the initialization to avoid the action limit time exceeding 1s.


* **Training Limits**: There is only baseline.py as a basic training object,so we have designed three effective training methods. The first method is astart vs baseline training until we can win the baseline. We use the second method Q-Approximate learning. Astart and baseline are used to train the required weights. Finally, after getting a better weight, the training speed will become particularly slow. Our third method uses adversarial training,and we take out the attacker and defense in Q-Approximate learning separately to train each other.

## Design Decisions

From project analysis above, we compared some AI methods below:

|     | Computation Time (In-Game) | Training Time | Ability of Generalization | Implementation Difficulty |
| --- | --- | --- | --- | --- |
| Astart search | short | - | - | easy |
| Model-based MDP | long - need long computation time to get the best policy for entire map | need more than one hour for each map | impossible to convert policy between maps | medium |
| Model-free MDP | long (need long computation time for tree search) | - | high | medium |
| Q-learning | short | long | low | easy |
| Q-approximate learning | short | medium | high | medium - need to carefully select features |
| Q-learning with neural network | short | extremely long, might take monthes | high if well trained | hard |


