class TicTacToe:
    def __init__(self):  # Corrected constructor name
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"  #human is x,ai is o
        self.game_over = False

    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * 5)

    def is_winner(self, player):
        b = self.board
        return any(all(b[i][j] == player for j in range(3)) for i in range(3)) or \
               any(all(b[i][j] == player for i in range(3)) for j in range(3)) or \
               all(b[i][i] == player for i in range(3)) or \
               all(b[i][2 - i] == player for i in range(3))

    def is_full(self):
        return all(cell != " " for row in self.board for cell in row)

    def minimax(self, is_maximizing):
        if self.is_winner("O"):
            return 1
        if self.is_winner("X"):
            return -1
        if self.is_full():
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == " ":
                        self.board[i][j] = "O"
                        score = self.minimax(False)
                        self.board[i][j] = " "
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == " ":
                        self.board[i][j] = "X"
                        score = self.minimax(True)
                        self.board[i][j] = " "
                        best_score = min(score, best_score)
            return best_score

    def best_move(self):
        best_score = -float('inf')
        move = (-1, -1)
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == " ":
                    self.board[i][j] = "O"
                    score = self.minimax(False)
                    self.board[i][j] = " "
                    if score > best_score:
                        best_score = score
                        move = (i, j)
        return move

    def play(self):
        while not self.game_over:
            self.print_board()

            if self.current_player == "X":
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter col (0-2): "))
                    if 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == " ":
                        self.board[row][col] = "X"
                    else:
                        print("Invalid move!")
                        continue
                except ValueError:
                    print("Enter valid numbers between 0 and 2.")
                    continue
            else:
                print("AI's turn (O)...")
                row, col = self.best_move()
                self.board[row][col] = "O"
                print(f"AI plays at ({row}, {col})")

            if self.is_winner(self.current_player):
                self.print_board()
                print(f"{'You' if self.current_player == 'X' else 'AI'} win!")
                self.game_over = True
            elif self.is_full():
                self.print_board()
                print("It's a draw!")
                self.game_over = True
            else:
                self.current_player = "O" if self.current_player == "X" else "X"


if __name__ == "__main__":
    game = TicTacToe()
    game.play()
