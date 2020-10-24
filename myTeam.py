# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, **kwargs):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    # return [transitionAttacktoDefense(firstIndex, **kwargs), transitionAttacktoDefense(secondIndex, **kwargs)]
    return [transitionAttacktoDefense(firstIndex, **kwargs), transitionAttacktoDefense(secondIndex, **kwargs)]


##########
# Agents #
##########

class ReinforcementAgent(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if kwargs.get('numTraining') != None:
            self.numTraining = int(kwargs['numTraining'])
        else:
            self.numTraining = 0

        self.episodeNumber = 0
        self.alpha = 0.  # learning rate
        self.discount = 0.9  # discount factor
        self.trainReward = 0.
        self.avoidSameAction = []


        self.weights = {'stop': -50.07039375227111, 'be-eaten': -748.7517951982985,
                        'go-home': 357.94288546135505, 'eats-food': 29.647654584354346,
                        'distance-ghost': 27.619043655219315, 'eat-capsule': 400,
                        'distance-capsule': -5.416031019651863, 'temp-go-home': 181.06855633391012,
                        'eat-ghost': 100, 'distance-food': -30.325076852073114, 'dist2bestentry': -22.442620798135035,
                        'in-dead-end': -57.06157636581033, 'returns-food': 357.94288546135505, 'bias': 0.00000001,'bias1':0.00000001}

    ######################
    ## Train funcations ##
    ######################

    def startTrain(self):
        self.preGameState = None
        self.preAction = None
        self.epiodeRewards = 0.
        self.start_time = time.time()

    def endTrain(self):
        if self.episodeNumber < self.numTraining:
            self.trainReward += self.epiodeRewards
        else:
            self.alpha = 0.
        self.episodeNumber += 1

    def observationFunction(self, gameState):
        if not self.preGameState is None:
            reward = self.getReward(gameState)
            self.processRecord(self.preGameState, self.preAction, gameState, reward)

        return gameState.makeObservation(self.index)

    def processRecord(self, gameState, action, nextGameState, reward):
        self.epiodeRewards += reward
        self.updateWeights(gameState, action, nextGameState, reward)

    def getReward(self, gameState):
        return gameState.getScore()

    def updateWeights(self, gameState, action, successorGameState, reward):
        features = self.getFeatures(gameState, action)
        correction = (reward + self.discount * self.computeValueFromQValues(successorGameState)) - self.evaluate(gameState, action)
        for feature, value in features.items():
            # correction = (reward + self.discount * self.computeValueFromQValues(successorGameState)) - self.evaluate(
            #     gameState, action)
            self.weights[feature] += self.alpha * correction * value

    ###############
    ### 开始训练 ###
    ###############
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startTrain()
        ####### ####### ####### 初始的时候食物一共多少的 ####### ####### ####### #######
        self.numberFood = self.getFood(gameState).asList()

        self.bestEntry = None
        self.ghostInfo = None
        self.tryForOffen = 0
        self.keepInMid = 0
        self.distance_normaliser = self.distanceNormaliser(gameState)
        self.initPosition = gameState.getInitialAgentPosition(self.index)
        self.offsafePosition = self.calculateDis2Entry(gameState)
        self.mazeProblem = 0
        self.totalFood = len(self.getFood(gameState).asList())
        entries = []
        for entry in self.countEntry(gameState):
            if entry[1] <= int(gameState.data.layout.height) and entry[1] >= int(gameState.data.layout.height / 3):
                entries.append(entry)
            if entry[1] <= int(gameState.data.layout.height / 3) and entry[1] >= 0:
                entries.append(entry)
        if self.index == 0 or self.index == 1:
            # print("possible entry for index 0", entries)
            self.farAwayMid = entries[0]
        else:
            # print("possible entry for index 2", entries)
            self.farAwayMid = entries[-1]


        # 下方是找死路及入口方法,得到5个list: self.deadlineFoods和self.deadlineFoodsPositions和self.deadendPositions和self.deadendEntry和self.deadlinePositions
        # self.deadlineFoods: [((x,y),depth), ...]  1. (x,y)--在死路里的food坐标; 2.food所在的死路深度(即从死路入口走多少步能到food)
        # self.deadlineFoodsPositions: [(x,y), ...] (x,y)--在死路里的food坐标
        # self.deadendPositions: [(x,y), ...]  (x,y)--死路点的坐标
        # self.deadendEntry: [(x,y), ...]  (x,y)--死路入口的坐标
        # self.deadlinePositions: [(x,y), ...]   (x,y)--死路里所有点的坐标

        self.walls = gameState.getWalls().asList()
        self.deadlineFoods = []
        self.deadlineFoodsPositions = []
        self.deadendPositions = []
        self.deadlineEntry = []
        self.deadlinePositions = []

        if self.index ==0 or self.index == 2:
            foods = [food for food in gameState.getBlueFood().asList()]
        else:
            foods = [food for food in gameState.getRedFood().asList()]

        def howManyWallsAround(x, y):
            howmanywallsaround = 0
            if (x + 1, y) in self.walls:
                howmanywallsaround += 1
            if (x - 1, y) in self.walls:
                howmanywallsaround += 1
            if (x, y + 1) in self.walls:
                howmanywallsaround += 1
            if (x, y - 1) in self.walls:
                howmanywallsaround += 1
            return howmanywallsaround

        def getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y):
            flagup = False
            flagdown = False
            flagleft = False
            flagright = False
            while howmanywallsaround > 1:
                # 判断情况，存在4种：只能up，只能down，只能left, 只能right
                flagup = False
                flagdown = False
                flagleft = False
                flagright = False

                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))

                if (flagvertical == True):
                    flagvertical = False
                    # 1. 只能up
                    if (x, y - 1) in self.walls:
                        y += 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                            y += 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagdown = True
                        flaghorizontal = True

                    # 2. 只能down
                    elif (x, y + 1) in self.walls:
                        y -= 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                            y -= 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagup = True
                        flaghorizontal = True

                elif (flaghorizontal == True):
                    flaghorizontal = False
                    # 3. 只能left
                    if (x + 1, y) in self.walls:
                        x -= 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                            x -= 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagright = True
                        flagvertical = True

                    # 4. 只能right
                    elif (x - 1, y) in self.walls:
                        x += 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                            x += 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagleft = True
                        flagvertical = True

                howmanywallsaround = howManyWallsAround(x, y)

            if (flagup == True):
                self.deadlineEntry.append((x, y + 1))
                self.deadlinePositions.append((x, y + 1))
            elif (flagdown == True):
                self.deadlineEntry.append((x, y - 1))
                self.deadlinePositions.append((x, y - 1))
            elif (flagleft == True):
                self.deadlineEntry.append((x - 1, y))
                self.deadlinePositions.append((x - 1, y))
            else:
                self.deadlineEntry.append((x + 1, y))
                self.deadlinePositions.append((x + 1, y))

            return depth

        lines = []  # 可以走的路的点
        deadlinepos1 = []  # 竖向下方死路点
        deadlinepos2 = []  # 竖向上方死路点
        deadlinepos3 = []  # 横向左方死路点
        deadlinepos4 = []  # 横向右方死路点

        layoutdata = self.walls[len(self.walls) - 1]
        layoutx = layoutdata[0]
        layouty = layoutdata[1]
        for x in range(0, layoutx):
            for y in range(0, layouty):
                if (x, y) not in self.walls:
                    lines.append((x, y))

        for (x, y) in lines:
            # 竖向下方死路点
            if (x - 1, y) in self.walls and (x + 1, y) in self.walls and (x, y - 1) in self.walls:
                deadlinepos1.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))
            # 竖向上方死路点
            if (x - 1, y) in self.walls and (x + 1, y) in self.walls and (x, y + 1) in self.walls:
                deadlinepos2.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))
            # 横向左方死路点
            if (x - 1, y) in self.walls and (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                deadlinepos3.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))
            # 横向右方死路点
            if (x + 1, y) in self.walls and (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                deadlinepos4.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))

        # 竖向下方死路点处理

        for (x, y) in deadlinepos1:
            deadlineFoodstemp = []
            depth = 0

            while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x, y + 1) > 1):
                    y += 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    y += 1
                    depth += 1
                    break;

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = True
                flagvertical = False
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        # 竖向上方死路点处理
        for (x, y) in deadlinepos2:
            deadlineFoodstemp = []
            depth = 0

            while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x, y - 1) > 1):
                    y -= 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    y -= 1
                    depth += 1
                    break;

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = True
                flagvertical = False
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        # 横向左方死路点处理
        for (x, y) in deadlinepos3:
            deadlineFoodstemp = []
            depth = 0
            while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x + 1, y) > 1):
                    x += 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    x += 1
                    depth += 1
                    break

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = False
                flagvertical = True
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        # 横向右方死路点处理
        for (x, y) in deadlinepos4:
            deadlineFoodstemp = []
            depth = 0
            while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x - 1, y) > 1):
                    x -= 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    x -= 1
                    depth += 1
                    break

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = False
                flagvertical = True
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        for (x, y) in self.deadlinePositions:
            if howManyWallsAround(x, y) <= 1:
                self.deadlinePositions.remove((x, y))


    ######################################################死路点处理结束######################################################

    def final(self, gameState):
        reward = self.getReward(gameState)
        # reward = self.epiodeRewards
        self.processRecord(self.preGameState, self.preAction, gameState, reward)
        self.endTrain()

        LIMIT_EPISODE= 10
        if self.episodeNumber % LIMIT_EPISODE == 0:
            print("Train state:",self.episodeNumber)
            print("The weight:",self.weights)
            print("Run Time",self.start_time - time.time())
            self.start_time = time.time()
            print('*'*30)
        if self.episodeNumber == self.numTraining:
            print('==' * 30)
            print('Training Done')
            print('Total Reward and Average Reward', self.epiodeRewards, self.epiodeRewards / self.numTraining)
            print("The score", gameState.getScore())
            print(self.getWeights())

    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self):
        return self.weights

    def evaluate(self, gameState, action):
        '''
        get the Q value
        '''
        features = self.getFeatures(gameState, action)
        weights = self.weights
        return features * weights

    def computeValueFromQValues(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) > 0:
            return max([self.evaluate(gameState, action) for action in legalActions])
        return 0

    def computeActionFromQValues(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        maxValue = self.computeValueFromQValues(gameState)
        bestActions = [action for action in legalActions if self.evaluate(gameState, action) == maxValue]
        if len(bestActions) > 0:
            return random.choice(bestActions)
        elif len(legalActions) > 0:
            return random.choice(legalActions)
        else:
            return None

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # def aStarSearchForSingleGoal(self, gameState, goal):
    #     """returns the the action as direction. to reach a certain goal, which is a tuple representing a location in the matrix"""
    #
    #     def state(node):
    #         # extract state from the node
    #         (someGameState, _, _) = node
    #         return someGameState.getAgentState(self.index).getPosition()
    #
    #     def g(node):
    #         # find the total cost to reach node
    #         (_, _, cost) = node
    #         return cost
    #
    #     def h(node):
    #         # use manhattan distance as heuristic for root planning
    #         # return util.manhattanDistance(state(node), goal)
    #         return self.getMazeDistance(state(node), goal)
    #
    #     def priority(node):
    #         # find the priority based on g and h
    #         return g(node) + h(node)
    #
    #     def extractActions(node):
    #         # extract actions from the node
    #         (_, actions, _) = node
    #         return actions
    #
    #     def isGoalState(node, goal):
    #         if state(node) == goal:
    #             return True
    #         return False
    #
    #     root_state = gameState
    #     root_node = (root_state, [], 0)  # a node contains (state, actions and cost to reach the state)
    #
    #     open_list = util.PriorityQueue()
    #     open_list.push(root_node, priority(root_node))
    #     closed = []
    #     best_g = {}  # dictionary mapping state to best_g
    #     succeed_stepCost = 1
    #
    #     while not open_list.isEmpty():
    #         node = open_list.pop()
    #
    #         if state(node) not in closed or (state(node) in best_g and g(node) < best_g.get(state(node))):
    #             # reopen for better g.
    #             # All nodes with same state and worse g are behind the node of the state with best g and will be skipped for their turn.
    #             closed = closed + [state(node)]
    #             best_g[state(node)] = g(node)
    #
    #             if isGoalState(node, goal):
    #                 return extractActions(node)
    #
    #             actions = node[0].getLegalActions(self.index)
    #
    #             for action in actions:
    #                 nextState = self.getSuccessor(node[0], action)
    #                 succeed_node = (nextState, extractActions(node) + [action], g(node) + succeed_stepCost)
    #                 if h(succeed_node) < float('inf'):
    #                     open_list.push(succeed_node, priority(succeed_node))
    #
    #     # print("no solution found for goal ", goal)
    #     return []

    # def aStarSearch(self, gameState, goals):
    #     "Returns the best action to reach to goal with min number of actions"
    #     # print("agent", self.index, gameState.getAgentState(self.index).getPosition(), goals)
    #     if len(goals) == 0:
    #         # print("WARNING NO GOAL specified! ")
    #         return random.choice(gameState.getLegalActions(self.index))
    #     actionsList = []
    #     bestActionsFound = False
    #     minPathLength = float("inf")
    #     # print(goals, gameState.getAgentState(self.index).getPosition(), self.index)
    #
    #     for goal in goals:
    #         actionsList.append(self.aStarSearchForSingleGoal(gameState, goal))
    #
    #     for actions in actionsList:
    #         if len(actions) > 0:
    #             if len(actions) < minPathLength:
    #                 minPathLength = len(actions)
    #                 bestActions = actions
    #                 bestActionsFound = True
    #     if (bestActionsFound):
    #         return bestActions[0]
    #     else:
    #         return random.choice(gameState.getLegalActions(self.index))

    def chooseAction(self, gameState):
        if self.getDefenders(gameState) is not None:
            # get Policy ---> best actions
            legalActions = gameState.getLegalActions(self.index)
            action = self.computeActionFromQValues(gameState)
            actionNow = action
            self.preGameState = gameState
            self.preAction = actionNow
            self.avoidSameAction.append(actionNow)
            return actionNow

    ############################## helper functions########################################################################
    def dist2NearestFood(self, gameState):
        foods = self.getFood(gameState).asList()
        maxScore = -999999
        maxScoreHelp = -999999
        SignmaxScore = -999999
        foodPosition = None
        attackAgentState = None
        help_attack_agent_state = None
        attackAgentPos = None
        help_attack_agent_pos = None

        if self.index == 0 or self.index == 2:
            attackAgentState = gameState.getAgentState(0)
            attackAgentPos = gameState.getAgentPosition(0)
            help_attack_agent_state = gameState.getAgentState(2)
            help_attack_agent_pos = gameState.getAgentPosition(2)
        elif self.index == 1 or self.index == 3:
            attackAgentState = gameState.getAgentState(1)
            attackAgentPos = gameState.getAgentPosition(1)
            help_attack_agent_state = gameState.getAgentState(3)
            help_attack_agent_pos = gameState.getAgentPosition(3)

        for f in foods:
            if attackAgentState.isPacman and help_attack_agent_state.isPacman:
                _, ghost_info = self.dist2NearestGhost(gameState)
                dis_attack_ghost = self.getMazeDistance(ghost_info.getPosition(),attackAgentPos) if ghost_info else 0
                dis_help_attack_ghost = self.getMazeDistance(ghost_info.getPosition(),help_attack_agent_pos) if ghost_info else 0
                dis_attack_food = self.getMazeDistance(f, attackAgentPos)
                dis_help_attack_food = self.getMazeDistance(f, help_attack_agent_pos)
                dis_two_attack_agent = self.getMazeDistance(attackAgentPos,help_attack_agent_pos)
                score_attack = dis_attack_ghost - dis_attack_food
                score_help_attack = dis_help_attack_ghost + dis_attack_food - dis_help_attack_food
                if score_attack > maxScore and (self.index == 0 or self.index == 1):
                    maxScore = score_attack
                    foodPosition = f
                if score_help_attack > maxScoreHelp and (self.index == 2 or self.index == 3):
                    maxScoreHelp = score_help_attack
                    foodPosition = f
            else:
                dis_ghost,_ = self.dist2NearestGhost(gameState)
                dis_ghost = dis_ghost if dis_ghost else 0
                dis_food = self.getMazeDistance(f,gameState.getAgentPosition(self.index))
                score = dis_ghost - dis_food
                if score > SignmaxScore:
                    SignmaxScore = score
                    foodPosition = f

        if foodPosition is None:
            return 0
        foodDistance = self.getMazeDistance(foodPosition, gameState.getAgentPosition(self.index))

        return foodDistance

    def dist2NearestCapsule(self, gameState):
        capsules = self.getCapsules(gameState)
        capsulesDistance = [self.getMazeDistance(gameState.getAgentPosition(self.index), capsule) for capsule in
                            capsules]
        if len(capsulesDistance) == 0:
            return 0
        return min(capsulesDistance)

    def closeCapsulePosition(self, gameState):
        capsules = self.getCapsules(gameState)
        capsulesDistance = [self.getMazeDistance(gameState.getAgentPosition(self.index), capsule) for capsule in
                            capsules]
        minCapDis = [mincap for mincap, d in zip(capsules, capsulesDistance) if d == min(capsulesDistance)]
        if len(minCapDis) == 0:
            return None
        return random.choice(minCapDis)

    # def closeFood(self, gameState):
    #
    #     foods = self.getFood(gameState).asList()
    #     foodDistance = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a) for a in foods]
    #     closeFood = [f for f, d in zip(foods, foodDistance) if d == min(foodDistance)]
    #     if len(closeFood) == 0:
    #         return None
    #     else:
    #         return random.choice(closeFood)

    def dist2NearestGhost(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostPosition = [opponent.getPosition() for opponent in opponents
                         if (opponent.getPosition())  # observable
                         and (not opponent.isPacman)  # is ghost
                         and (opponent.scaredTimer < 1)  # not scared
                         ]
        ghostDistance = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), ghostPos) for ghostPos
                         in
                         ghostPosition]
        ghostInfo = [opponent for opponent in opponents
                     if (opponent.getPosition())  # observable
                     and (not opponent.isPacman)  # is ghost
                     ]
        minghostInfo = [g for g, d in zip(ghostInfo, ghostDistance) if d == min(ghostDistance)]
        if len(ghostDistance) == 0:
            return None, None
        if max(ghostDistance) >= 6:
            return None, None
        return min(ghostDistance), random.choice(minghostInfo)

    def dist2NearestHome(self, gameState):
        homeEntrys = self.countEntry(gameState)
        homeDistance = [self.getMazeDistance(gameState.getAgentPosition(self.index), homeEntry) for homeEntry in
                        homeEntrys]
        if len(homeDistance) == 0:
            return None
        return min(homeDistance)
    def comeBack(self,gameState):
        if self.index ==0 or self.index == 2:
            mindlineX = int(gameState.data.layout.width / 2) - 1
        else:
            mindlineX = int(gameState.data.layout.width / 2)

        boundary = [(mindlineX, y) for y in range(0, gameState.data.layout.height)]
        positionEntrys = [x for x in boundary if x not in gameState.getWalls().asList()]
        homeDistance = [self.getMazeDistance(gameState.getAgentPosition(self.index), homeEntry) for homeEntry in
                        positionEntrys]
        if len(homeDistance) == 0:
            return None
        return min(homeDistance)


    def dist2NearestHomep(self, gameState):
        homeEntrys = self.countEntryforBack(gameState)
        homeDistance = [self.getMazeDistance(gameState.getAgentPosition(self.index), homeEntry) for homeEntry in
                        homeEntrys]
        position = [p for p, d in zip(homeEntrys, homeDistance) if d == min(homeDistance)]

        if len(position) == 0:
            return None
        return random.choice(position)

    def distanceNormaliser(self, gameState):

        return gameState.getWalls().count(item=False)

    def countEntry(self, gameState):

        if self.index ==0 or self.index ==2:
            mindlineX = int(gameState.data.layout.width / 2)
        else:
            mindlineX = int(gameState.data.layout.width / 2) - 1

        boundary = [(mindlineX, y) for y in range(0, gameState.data.layout.height)]
        positionEntrys = [x for x in boundary if x not in gameState.getWalls().asList()]
        return positionEntrys

    def countEntryforBack(self, gameState):

        if self.index == 0 or self.index == 2:
            mindlineX = int(gameState.data.layout.width / 2) - 1
        else:
            mindlineX = int(gameState.data.layout.width / 2)

        boundary = [(mindlineX, y) for y in range(0, gameState.data.layout.height)]
        positionEntrys = [x for x in boundary if x not in gameState.getWalls().asList()]
        return positionEntrys

    def calculateDis2Entry(self, gameState):

        entrys = self.countEntry(gameState)
        minDist2Entries = 10000000
        result = None
        walls = gameState.getWalls()
        if self.index ==0 or self.index == 2:
            mindlineX = int(gameState.data.layout.width / 2) - 1
            X = 0
            Y = mindlineX
        else:
            mindlineX = int(gameState.data.layout.width / 2)
            X = mindlineX
            Y = int(gameState.data.layout.width)
        for x in range(X, Y):
            for y in range(0, int(gameState.data.layout.height)):
                if not walls[x][y]:
                    dis2Entry = sum(self.getMazeDistance((x, y), entry) for entry in entrys)
                    if dis2Entry < minDist2Entries:
                        minDist2Entries = dis2Entry
                        result = (x, y)

        return result

    def calculateMinDis2Entry(self, gameState, myPos):

        dis = []
        entrys = self.countEntry(gameState)  # 需要防守的入口
        for entry in entrys:
            dis.append(self.getMazeDistance(myPos, entry))

        closeEntry = [c for c, d in zip(entrys, dis) if d == min(dis)]

        return random.choice(closeEntry)

    def getDefenders(self, gameState):

        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        return defenders

    def dist2GhostAndEntry(self, gameState):

        entrys = self.countEntry(gameState)
        ghostEntryDistance = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), entry) for
                              entry
                              in entrys]
        if len(ghostEntryDistance) == 0:
            return 0
        return min(ghostEntryDistance)

    def boundaryofRedandBlue(self, gameState):

        if self.index ==0 or self.index ==2:
            boundary = [((gameState.data.layout.width / 2) - 1, y) for y in range(0, gameState.data.layout.height)]
        else:
            boundary = [((gameState.data.layout.width / 2), y) for y in range(0, gameState.data.layout.height)]

        boundarywithoutwall = [x for x in boundary if x not in gameState.getWalls().asList()]

        return boundarywithoutwall

    def isDeadlineEntry(self, myPos):
        if myPos in self.deadlineEntry:
            return True
        else:
            return False

    def isInDeadline(self, myPos):
        if myPos in self.deadlinePositions:
            return True
        else:
            return False

    def checkLoopAction(self, gameState):
        if len(self.avoidSameAction) < 4:
            return None
        if self.avoidSameAction[-1] == self.avoidSameAction[-3] and self.avoidSameAction[-2] == self.avoidSameAction[-4] \
                and self.avoidSameAction[-1] != self.avoidSameAction[-2] and self.avoidSameAction[-3] != \
                self.avoidSameAction[-4]:
            loopAction = self.avoidSameAction[-2]
            self.avoidSameAction = []
            return loopAction
        else:
            return None

class OffensiveAgent(ReinforcementAgent):

    def chooseAction(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        action = self.computeActionFromQValues(gameState)

        loopAction = self.checkLoopAction(gameState)
        if loopAction is not None and loopAction in legalActions:
            legalActions.remove(loopAction)
            return random.choice(legalActions)
        self.preGameState = gameState
        self.preAction = action
        self.avoidSameAction.append(action)
        return action

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        newPosition = successor.getAgentPosition(self.index)
        myPos = gameState.getAgentPosition(self.index)
        moveleft = int(gameState.data.timeleft) / 4
        capsules = self.getCapsules(gameState)

        cur_distance_ghost, cur_ghost_info = self.dist2NearestGhost(gameState)
        next_distance_ghost, next_ghost_info = self.dist2NearestGhost(successor)

        cur_distance_ghost = cur_distance_ghost if cur_distance_ghost else 6
        next_distance_ghost = next_distance_ghost if next_distance_ghost else 6

        cur_distance_food = self.dist2NearestFood(gameState)
        next_distance_food = self.dist2NearestFood(successor)

        cur_distance_home = self.dist2NearestHome(gameState)
        # next_distance_home = self.dist2NearestHome(successor)

        cur_distance_capsule = self.dist2NearestCapsule(gameState)
        next_distance_capsule = self.dist2NearestCapsule(successor)

        if action == Directions.STOP: features['stop'] = 1
        if newPosition == self.initPosition and gameState.getAgentState(self.index).isPacman:
            features['be-eaten'] = 1

        if moveleft - 10 <= cur_distance_home: features['go-home'] = cur_distance_home / self.distance_normaliser

        if not successor.getAgentState(self.index).isPacman and gameState.getAgentState(
                self.index).numCarrying > 0:
            features['returns-food'] = 1.
        # 敌方区域
        if gameState.getAgentState(self.index).isPacman:

            next_distance_home = self.dist2NearestHome(successor)
            if len(self.getFood(gameState).asList()) <= 2:
                features['go-home'] = -next_distance_home / self.distance_normaliser

            else:
                if gameState.getAgentState(self.index).numCarrying <= int(self.totalFood - 2 / 3):
                    features['distance-food'] = float(next_distance_food) / self.distance_normaliser
                    features['distance-ghost'] = float(next_distance_ghost) / self.distance_normaliser
                    features['distance-capsule'] = float(next_distance_capsule) / self.distance_normaliser
                    features['bias'] = -float(cur_distance_food - next_distance_food) / self.distance_normaliser
                    if next_distance_ghost == 6:
                        features['bias1'] = -float(cur_distance_home - next_distance_home) / self.distance_normaliser
                else:
                    home_position = self.dist2NearestHomep(gameState)
                    tnext_distance_home = self.getMazeDistance(successor.getAgentPosition(self.index), home_position)
                    features['temp-go-home'] = -tnext_distance_home / self.distance_normaliser

            if len(capsules) > 0 and cur_distance_ghost <= 5:
                if newPosition in capsules:
                    features['eat-capsule'] = 1

            isfood = self.getFood(gameState)
            if not (next_ghost_info and next_ghost_info.scaredTimer <= 10):
                if (isfood[int(newPosition[0])][int(newPosition[1])]):
                    features['eats-food'] = 1

            if next_ghost_info and next_ghost_info.scaredTimer <= 10:
                features['distance-ghost'] = next_distance_ghost / self.distance_normaliser
                if next_distance_ghost and next_distance_ghost <= 1:
                    features['be-eaten'] = 1

                if self.isInDeadline(newPosition):
                    # print("can not enter into dead end")
                    features['in-dead-end'] = 1
                if self.isInDeadline(myPos):
                    features['be-eaten'] = 1
                    for capsule in capsules:
                        if capsule in self.deadlinePositions:
                            features['distance-ghost'] = next_distance_ghost / self.distance_normaliser
                if len(capsules) > 0:
                    if newPosition in capsules:
                        features['eat-capsule'] = 1
                    else:
                        next_distance_capsule = self.dist2NearestCapsule(successor)
                        features['distance-capsule'] = next_distance_capsule / self.distance_normaliser

                        if cur_distance_ghost <= 2:
                            if (isfood[int(newPosition[0])][int(newPosition[1])]):
                                features['eats-food'] = 0
                        if cur_distance_ghost <= 3:
                            features['distance-food'] = 0
                    features['bias'] = -float(cur_distance_capsule - next_distance_capsule) / self.distance_normaliser

                else:
                    # 逃跑回家路线判断

                    if successor.getAgentState(self.index).getPosition() == gameState.getInitialAgentPosition(
                            self.index) and gameState.getAgentState(self.index).isPacman:
                        features['be-eaten'] = 1
                    if next_distance_ghost <= 1:
                        features['be-eaten'] = 1
                    if self.isInDeadline(newPosition):
                        features['in-dead-end'] = 1
                    if self.isInDeadline(myPos):
                        # 本身在死路里面，需要快点出来
                        features['distance-ghost'] = -cur_distance_ghost / self.distance_normaliser
                        features['in-dead-end'] = 1

                    if len(self.getFood(gameState).asList()) <= 2:
                        features['go-home'] = -next_distance_home / self.distance_normaliser
                    else:
                        diff = cur_distance_home - next_distance_home
                        # ghost_to_home = next_distance_ghost - next_distance_home
                        features['bias'] = -diff / self.distance_normaliser

                        home_position = self.dist2NearestHomep(gameState)
                        tnext_distance_home = self.getMazeDistance(gameState.getAgentPosition(self.index),
                                                                   home_position)
                        features['temp-go-home'] = -tnext_distance_home / self.distance_normaliser

                        if cur_distance_ghost <= 4:
                            if (isfood[int(newPosition[0])][int(newPosition[1])]):
                                features['eats-food'] = 0
                            features['distance-food'] = 0

                if next_ghost_info.scaredTimer > 0 and newPosition == next_ghost_info.getPosition():
                    features['eat-ghost'] = 1

            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            defences = [a for a in enemies if not a.isPacman and a.getPosition() != None]

            if len(defences) >= 2:
                home_position = self.dist2NearestHomep(gameState)
                tnext_distance_home = self.getMazeDistance(successor.getAgentPosition(self.index), home_position)
                features['temp-go-home'] = -tnext_distance_home / self.distance_normaliser
                all_ghost_dis = 0
                for defence in defences:
                    dis = self.getMazeDistance(successor.getAgentPosition(self.index), defence.getPosition())
                    all_ghost_dis += dis
                features['distance-ghost'] = all_ghost_dis / self.distance_normaliser

        else:
            food_list = []

            all_food = self.getFood(gameState).asList()
            for (x, y) in all_food:
                if self.index == 0 or self.index == 1:
                    if y > gameState.data.layout.height / 2:
                        food_list.append((x, y))
                else:
                    if y <= gameState.data.layout.height / 2:
                        food_list.append((x, y))
            if len(food_list) == 0:
                food_list = all_food
            entries = self.countEntry(gameState)
            h = int(gameState.data.layout.height / 2)
            l = int(gameState.data.layout.width / 2)

            if cur_ghost_info:
                dist_be_eaten = self.getMazeDistance(newPosition, cur_ghost_info.getPosition())
                if dist_be_eaten <= 1 and successor.getAgentState(self.index).isPacman:
                    features['be-eaten'] = 1
            if next_ghost_info:
                self.ghostInfo = next_ghost_info
                self.tryForOffen += 1
                if self.index == 0 or self.index == 2:
                    if newPosition[1] in [h, h + 1, h - 1] and newPosition[0] >= l - 4:
                        self.keepInMid += 1
                else:
                    if newPosition[1] in [h, h + 1, h - 1] and newPosition[0] <= l + 4:
                        self.keepInMid += 1
            maxscore = -10000000
            for entry in entries:
                dis2ghost = self.getMazeDistance(self.ghostInfo.getPosition(), entry) if self.ghostInfo else 0
                if (len(self.getFood(gameState).asList())) > 0:
                    dis2Food = min([self.getMazeDistance(entry, f) for f in food_list])
                    score = dis2ghost - dis2Food
                    if score > maxscore:
                        maxscore = score
                        self.bestEntry = entry
                else:
                    self.bestEntry = self.offsafePosition
            # print("入口",self.bestEntry)
            features['dist2bestentry'] = self.getMazeDistance(newPosition,
                                                              self.bestEntry) / self.distance_normaliser
            if self.tryForOffen >= 30:
                # print("换路")
                features['dist2bestentry'] = self.getMazeDistance(newPosition,
                                                                  self.offsafePosition) / self.distance_normaliser
            if self.keepInMid >= 30:
                # print("换路",self.bestEntry)
                features['dist2bestentry'] = self.getMazeDistance(newPosition,
                                                                  self.farAwayMid) / self.distance_normaliser
            if self.mazeProblem >= 5:
                features['dist2bestentry'] = self.getMazeDistance(newPosition,
                                                                  self.offsafePosition) / self.distance_normaliser

            if newPosition == self.offsafePosition:
                self.tryForOffen = 0
                self.mazeProblem = 0
            if newPosition == self.farAwayMid:
                self.keepInMid = 0

            bias = self.getMazeDistance(newPosition,
                                        gameState.getInitialAgentPosition(self.index)) / self.distance_normaliser
            features['bias'] = bias

        return features

    def getReward(self, gameState):

        curX, curY = gameState.getAgentState(self.index).getPosition()

        curX = int(curX)
        curY = int(curY)
        reward = 0
        preState = self.preGameState
        preX, preY = preState.getAgentState(self.index).getPosition()
        preX = int(preX)
        preY = int(preY)
        prevFood = self.getFood(self.preGameState)
        prevCapsule = self.getCapsules(self.preGameState)

        if prevFood[curX][curY]:
            reward += 1
        if (curX, curY) in prevCapsule:
            reward += 10
        # 被吃了
        if (curX, curY) == gameState.getInitialAgentPosition(self.index) and preState.getAgentState(
                self.index).isPacman and not gameState.getAgentState(self.index).isPacman:
            reward -= 100
        # Stop
        if (curX, curY) == (preX, preY):
            reward -= 1
        if preState.getAgentState(self.index).numCarrying < gameState.getAgentState(self.index).numCarrying and \
                (curX, curY) != gameState.getInitialAgentPosition(self.index):
            reward += 4 * gameState.getAgentState(self.index).numCarrying
        reward += gameState.getScore() - self.preGameState.getScore()

        return reward


class DefenseAgent(ReinforcementAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.preGameState = None
        self.preAction = None
        self.startTrain()
        self.flag = False
        self.weights = {'onDefense': 1000,
                        'numInvaders': -992.468250863138,
                        'invaderDistance': -9.403164039542512,
                        'stop': -100, 'reverse': -7.987298287555759,
                        'toSecFood': -3.565494519727313, 'defencePosition': -100.20305713803033,
                        'eat-invader': 1000, 'bias': 0.00001,'go-home':357.94288546135505,'distance-ghost':27,'in-dead-end':-888
                        ,'far-away-each-other':10}  # 'to-other-agent-ghost':-10

        self.totalFood = self.getFood(gameState).asList()
        self.taget = None
        self.safePosition = self.calculateDis2Entry(gameState)
        self.need_double_denfenses = None
        self.distance_normaliser = self.distanceNormaliser(gameState)

        self.walls = gameState.getWalls().asList()
        self.deadlineFoods = []
        self.deadlineFoodsPositions = []
        self.deadendPositions = []
        self.deadlineEntry = []
        self.deadlinePositions = []

        if self.index == 0 or self.index == 2:
            foods = [food for food in gameState.getBlueFood().asList()]
        else:
            foods = [food for food in gameState.getRedFood().asList()]

        def howManyWallsAround(x, y):
            howmanywallsaround = 0
            if (x + 1, y) in self.walls:
                howmanywallsaround += 1
            if (x - 1, y) in self.walls:
                howmanywallsaround += 1
            if (x, y + 1) in self.walls:
                howmanywallsaround += 1
            if (x, y - 1) in self.walls:
                howmanywallsaround += 1
            return howmanywallsaround

        def getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y):
            flagup = False
            flagdown = False
            flagleft = False
            flagright = False
            while howmanywallsaround > 1:
                # 判断情况，存在4种：只能up，只能down，只能left, 只能right
                flagup = False
                flagdown = False
                flagleft = False
                flagright = False

                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))

                if (flagvertical == True):
                    flagvertical = False
                    # 1. 只能up
                    if (x, y - 1) in self.walls:
                        y += 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                            y += 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagdown = True
                        flaghorizontal = True

                    # 2. 只能down
                    elif (x, y + 1) in self.walls:
                        y -= 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                            y -= 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagup = True
                        flaghorizontal = True

                elif (flaghorizontal == True):
                    flaghorizontal = False
                    # 3. 只能left
                    if (x + 1, y) in self.walls:
                        x -= 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                            x -= 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagright = True
                        flagvertical = True

                    # 4. 只能right
                    elif (x - 1, y) in self.walls:
                        x += 1
                        depth += 1
                        self.deadlinePositions.append((x, y))
                        while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                            x += 1
                            depth += 1
                            self.deadlinePositions.append((x, y))
                        flagleft = True
                        flagvertical = True

                howmanywallsaround = howManyWallsAround(x, y)

            if (flagup == True):
                self.deadlineEntry.append((x, y + 1))
                self.deadlinePositions.append((x, y + 1))
            elif (flagdown == True):
                self.deadlineEntry.append((x, y - 1))
                self.deadlinePositions.append((x, y - 1))
            elif (flagleft == True):
                self.deadlineEntry.append((x - 1, y))
                self.deadlinePositions.append((x - 1, y))
            else:
                self.deadlineEntry.append((x + 1, y))
                self.deadlinePositions.append((x + 1, y))

            return depth

        lines = []  # 可以走的路的点
        deadlinepos1 = []  # 竖向下方死路点
        deadlinepos2 = []  # 竖向上方死路点
        deadlinepos3 = []  # 横向左方死路点
        deadlinepos4 = []  # 横向右方死路点

        layoutdata = self.walls[len(self.walls) - 1]
        layoutx = layoutdata[0]
        layouty = layoutdata[1]
        for x in range(0, layoutx):
            for y in range(0, layouty):
                if (x, y) not in self.walls:
                    lines.append((x, y))

        for (x, y) in lines:
            # 竖向下方死路点
            if (x - 1, y) in self.walls and (x + 1, y) in self.walls and (x, y - 1) in self.walls:
                deadlinepos1.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))
            # 竖向上方死路点
            if (x - 1, y) in self.walls and (x + 1, y) in self.walls and (x, y + 1) in self.walls:
                deadlinepos2.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))
            # 横向左方死路点
            if (x - 1, y) in self.walls and (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                deadlinepos3.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))
            # 横向右方死路点
            if (x + 1, y) in self.walls and (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                deadlinepos4.append((x, y))
                self.deadendPositions.append((x, y))
                self.deadlineEntry.append((x, y))
                self.deadlinePositions.append((x, y))

        # 竖向下方死路点处理

        for (x, y) in deadlinepos1:
            deadlineFoodstemp = []
            depth = 0

            while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x, y + 1) > 1):
                    y += 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    y += 1
                    depth += 1
                    break;

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = True
                flagvertical = False
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        # 竖向上方死路点处理
        for (x, y) in deadlinepos2:
            deadlineFoodstemp = []
            depth = 0

            while (x - 1, y) in self.walls and (x + 1, y) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x, y - 1) > 1):
                    y -= 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    y -= 1
                    depth += 1
                    break;

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = True
                flagvertical = False
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        # 横向左方死路点处理
        for (x, y) in deadlinepos3:
            deadlineFoodstemp = []
            depth = 0
            while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x + 1, y) > 1):
                    x += 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    x += 1
                    depth += 1
                    break

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = False
                flagvertical = True
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        # 横向右方死路点处理
        for (x, y) in deadlinepos4:
            deadlineFoodstemp = []
            depth = 0
            while (x, y + 1) in self.walls and (x, y - 1) in self.walls:
                if ((x, y)) in foods:
                    foods.remove((x, y))
                    deadlineFoodstemp.append(((x, y), depth))
                if (howManyWallsAround(x - 1, y) > 1):
                    x -= 1
                    depth += 1
                    self.deadlinePositions.append((x, y))
                else:
                    x -= 1
                    depth += 1
                    break

            howmanywallsaround = howManyWallsAround(x, y)

            if (howmanywallsaround > 1):
                flaghorizontal = False
                flagvertical = True
                depth = getDeadlineEntry(howmanywallsaround, flagvertical, flaghorizontal, depth, x, y)

            for (x, y), curdepth in deadlineFoodstemp:
                self.deadlineFoods.append(((x, y), depth - curdepth))
                self.deadlineFoodsPositions.append((x, y))

        for (x, y) in self.deadlinePositions:
            if howManyWallsAround(x, y) <= 1:
                self.deadlinePositions.remove((x, y))
        if self.index == 0 or self.index == 2:
            self.allIndex = gameState.getRedTeamIndices()
            self.otherAgentIndex = [x for x in self.allIndex if x != self.index]
            self.otherAgentIndex = self.otherAgentIndex[0]
        else:
            self.allIndex = gameState.getBlueTeamIndices()
            self.otherAgentIndex = [x for x in self.allIndex if x != self.index]
            self.otherAgentIndex = self.otherAgentIndex[0]
    def chooseAction(self, gameState):

        # get Policy ---> best actions
        legalActions = gameState.getLegalActions(self.index)
        action = self.computeActionFromQValues(gameState)

        self.preGameState = gameState
        self.preAction = action

        return action

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        otherState = successor.getAgentState(self.otherAgentIndex)
        myPos = myState.getPosition()
        otherPos = otherState.getPosition()
        myCurPos = gameState.getAgentPosition(self.index)
        if self.index ==0 or self.index ==2:
            mindlineX = int(gameState.data.layout.width / 2) - 1
        else:
            mindlineX = int(gameState.data.layout.width / 2)

        if gameState.getAgentState(self.index).isPacman:
            next_distance_home = self.comeBack(successor)
            features['go-home'] = -next_distance_home / self.distance_normaliser
            next_distance_ghost, next_ghost_info = self.dist2NearestGhost(successor)
            cur_distance_ghost, cur_ghost_info = self.dist2NearestGhost(gameState)
            next_distance_ghost = next_distance_ghost if next_distance_ghost else 6
            cur_distance_ghost = cur_distance_ghost if cur_distance_ghost else 6
            features['distance-ghost'] = float(next_distance_ghost) / self.distance_normaliser
            if self.isDeadlineEntry(myPos):
                features['in-dead-end'] = 1
            if myPos in self.deadlinePositions:
                features['distance-ghost'] = -cur_distance_ghost / self.distance_normaliser
                features['in-dead-end'] = 1
        else:
            # Computes whether we're on defense (1) or offense (0)
            features['onDefense'] = 1
            if myState.isPacman:
                features['onDefense'] = 0

            # Computes distance to invaders we can see
            enemies = [successor.getAgentState(i)
                       for i in self.getOpponents(successor)]
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

            features['numInvaders'] = len(invaders) / 10

            if not myState.isPacman and not otherState.isPacman and invaders != None:
                dist = self.getMazeDistance(myPos,otherPos)
                features['far-away-each-other'] = dist / self.distance_normaliser
            if len(invaders) > 0:
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]

                features['invaderDistance'] = min(dists) / self.distance_normaliser

                for a in invaders:
                    if self.index ==0 or self.index ==2 :
                        if min(dists) == 0 and a.getPosition()[0] <= mindlineX:
                            features['eat-invader'] = 1
                            self.taget = None
                    else:
                        if min(dists) == 0 and a.getPosition()[0] >= mindlineX:
                            features['eat-invader'] = 1
                            self.taget = None
            else:
                if self.preGameState:
                    preFood = self.getFoodYouAreDefending(self.preGameState).asList()
                    curFood = self.getFoodYouAreDefending(gameState).asList()

                    foodBeEaten = [x for x in preFood if x not in curFood]
                    foodDistance = 999999
                    if len(foodBeEaten) > 0:
                        for f in curFood:
                            dist = self.getMazeDistance(f, foodBeEaten[0])
                            if dist < foodDistance:
                                self.taget = f
                                foodDistance = dist

                if self.taget is None:
                    features['defencePosition'] = self.getMazeDistance(myPos, self.safePosition) / self.distance_normaliser
                else:
                    features['toSecFood'] = self.getMazeDistance(self.taget, myPos) / self.distance_normaliser
                    bias = self.getMazeDistance(myPos, self.safePosition) - self.getMazeDistance(self.taget, myPos)
                    features['bias'] = bias / self.distance_normaliser

                if myCurPos == self.taget:
                    self.taget = None

        if action == Directions.STOP:
            features['stop'] = 1 / 10

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1 / 10

        return features

    def getReward(self, gameState):
        curX, curY = gameState.getAgentState(self.index).getPosition()
        curX = int(curX)
        curY = int(curY)
        reward = 0

        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]

        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        dists = [self.getMazeDistance((curX, curY), a.getPosition()) for a in invaders]

        if len(invaders) > 0:
            if min(dists) == 0 and gameState.getAgentState(self.index).scaredTimer == 0:
                reward += 10
        return reward

class transitionAttacktoDefense(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.attack = OffensiveAgent(*args, **kwargs)
        self.defense = DefenseAgent(*args, **kwargs)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.attack.registerInitialState(gameState)
        self.defense.registerInitialState(gameState)
        self.totalFood = len(self.getFoodYouAreDefending(gameState).asList())
        self.clostGhost = None
        # get the index of other agent as list
        if self.index == 0 or self.index == 2:
            self.allIndex = gameState.getRedTeamIndices()
            self.otherAgentIndex = [x for x in self.allIndex if x != self.index]
            self.otherAgentIndex = self.otherAgentIndex[0]
        else:
            self.allIndex = gameState.getBlueTeamIndices()
            self.otherAgentIndex = [x for x in self.allIndex if x != self.index]
            self.otherAgentIndex = self.otherAgentIndex[0]

    def chooseAction(self, gameState):
        myState = gameState.getAgentState(self.index)
        max_invader_numberCarrying = 0
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        curFood = self.getFoodYouAreDefending(gameState).asList()
        for invader in invaders:
            if invader.numCarrying > max_invader_numberCarrying:
                max_invader_numberCarrying = invader.numCarrying
            if max_invader_numberCarrying > 10 or len(curFood) <= 6:
                return self.defense.chooseAction(gameState)

            dist = self.getMazeDistance(myState.getPosition(), invader.getPosition())
            if not myState.isPacman and dist <= 5 and myState.scaredTimer < 1:
                return self.defense.chooseAction(gameState)

        if len(curFood) <= 12 and self.index == self.getTeam(gameState)[1]:
            return self.defense.chooseAction(gameState)
        if self.index == 0 or self.index == 2:

            if self.index == 0:
                return self.attack.chooseAction(gameState)
            if self.index == 2:
                return self.attack.chooseAction(gameState)
        else:
            # curFood = self.getFoodYouAreDefending(gameState).asList()
            if self.index == 1:
                return self.attack.chooseAction(gameState)
            if self.index == 3:
                return self.attack.chooseAction(gameState)