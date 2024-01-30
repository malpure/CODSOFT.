import math
import random

def print_board(board):
   
    for row in board:
        print(" ".join(row))
    print()

def is_winner(board, player):
    
    for i in range(3):
        
        if all(cell == player for cell in board[i]) or all(board[j][i] == player for j in range(3)):
            return True

    
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

def is_board_full(board):
   
    return all(cell != ' ' for row in board for cell in row)

def get_empty_cells(board):
   
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

def minimax(board, depth, maximizing_player, alpha, beta):
    
    if is_winner(board, 'O'):
        return 1
    elif is_winner(board, 'X'):
        return -1
    elif is_board_full(board):
        return 0

    if maximizing_player:
        max_eval = -math.inf
        for i, j in get_empty_cells(board):
            board[i][j] = 'O'
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[i][j] = ' '
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for i, j in get_empty_cells(board):
            board[i][j] = 'X'
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[i][j] = ' '
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board):
    
    best_val = -math.inf
    best_move = (-1, -1)
    for i, j in get_empty_cells(board):
        board[i][j] = 'O'
        move_val = minimax(board, 0, False, -math.inf, math.inf)
        board[i][j] = ' '
        if move_val > best_val:
            best_move = (i, j)
            best_val = move_val
    return best_move

def play_tic_tac_toe():
    
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    while True:
        print_board(board)

        if current_player == 'X':
           
            row = int(input("Enter row (0, 1, or 2): "))
            col = int(input("Enter column (0, 1, or 2): "))
            if board[row][col] != ' ':
                print("Cell already occupied. Try again.")
                continue
        else:
           
            row, col = get_best_move(board)
            print(f"AI plays at row {row}, column {col}")

       
        board[row][col] = current_player

        if is_winner(board, current_player):
            
            print_board(board)
            print(f"{current_player} wins!")
            break
        elif is_board_full(board):
            
            print_board(board)
            print("It's a tie!")
            break

        
        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    
    play_tic_tac_toe()
