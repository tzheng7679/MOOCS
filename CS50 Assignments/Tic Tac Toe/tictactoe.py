"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    xCount = 0;
    oCount = 0;

    for row in range(0, 3):
        for col in range(0, 3):
            if board[row][col] == X: 
                xCount += 1
            if board[row][col] == O:
                oCount += 1
    if xCount == oCount:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set({})

    for row in range(0, 3):
        for col in range(0, 3):
            if board[row][col] == EMPTY:
                moves.add((row, col))
    
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    isFound = False
    for thing in actions(board):
        if action == thing:
            isFound = True
            break
    if not isFound:
        raise NotImplementedError

    newBoard = copy.deepcopy(board)
    newBoard[action[0]][action[1]] = player(board)
    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    #check for rows
    for row in range(0, 3):
        if(board[row][0] == board[row][1] and board[row][1] == board[row][2] and board[row][0] != EMPTY):
            return board[row][0]
    #chekc for cols
    for col in range(0, 3):
        if(board[0][col] == board[1][col] and board[1][col] == board[2][col] and board[0][col] != EMPTY):
            return board[0][col]
    
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]
    
    if board[0][2] == board[1][1] and board[2][0] == board[1][1] and board[0][2] != EMPTY:
        return board[2][0]
    
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None or len(actions(board)) == 0:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    thePlayer = player(board)

    acts = actions(board)
    maxScore = 0
    if thePlayer == O:
        maxScore = 2
    if thePlayer == X:
        maxScore = -2

    bestMove = None
    
    for act in acts:
        if terminal(result(board, act)):
            score = utility(result(board, act))
            if thePlayer == O:
                if score == -1:
                    return act
                if score < maxScore:
                    maxScore = score
                    bestMove = act
            else:
                if score == 1:
                    return act
                if score > maxScore:
                    maxScore = score
                    bestMove = act
        else:
            newBoard = result(board, act)
            while terminal(newBoard) == False:
                newMove = minimax(newBoard)
                newBoard = result(newBoard, newMove)

            score = utility(newBoard)
            if thePlayer == O:
                if score == -1:
                    return act
                if score < maxScore:
                    maxScore = score
                    bestMove = act
            else:
                if score == 1:
                    return act
                if score > maxScore:
                    maxScore = score
                    bestMove = act
    return bestMove

