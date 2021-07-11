import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""
def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]                
        
        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myBreadthFirstSearch(problem):
    visited = {}
    frontier = util.Queue()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myAStarSearch(problem, heuristic):
    visited = {}
    frontier = util.PriorityQueue()

    frontier.push((problem.getStartState(), None, 0), heuristic(problem.getStartState()))

    while not frontier.isEmpty():
        state, prev_state, f_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state, f_state+step_cost), f_state+step_cost+heuristic(next_state))

    return []
"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""
class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if state.isTerminated() or depth == 0:
            return None, state.evaluateScore()        

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        for child in state.getChildren():

            if child.isMe():
                _, score = self.minimax(child, depth - 1)
            else:
                _, score = self.minimax(child, depth)

            if state.isMe() and score > best_score:
                best_score = score
                best_state = child
            elif not state.isMe() and score < best_score:
                best_score = score
                best_state = child
        
        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state

class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth

    def alphabeta(self, state, depth, alpha, beta):
        if state.isTerminated() or depth == 0:
            return None, state.evaluateScore()

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        for child in state.getChildren():

            if child.isMe():
                _, score = self.alphabeta(child, depth - 1, alpha, beta)
            else:
                _, score = self.alphabeta(child, depth, alpha, beta)

            if state.isMe():
                if score > beta:
                    return child, score
                if score > alpha:
                    alpha = score
                if score > best_score:
                    best_score = score
                    best_state = child
            else:
                if score < alpha:
                    return child, score
                if score < beta:
                    beta = score
                if score < best_score:
                    best_score = score
                    best_state = child

        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.alphabeta(state, self.depth, -float('inf'), float('inf'))
        return best_state