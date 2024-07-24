import itertools
import random
import copy


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return copy.deepcopy(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return copy.deepcopy(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count = self.count - 1



    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        self.cells.discard(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        print(str(cell) + " is a mine!")
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        self.moves_made.add(cell)
        self.mark_safe(cell)

        h = self.height
        w = self.width

        i = cell[0]
        j = cell[1]

        cells = set()

        for row in range(i - 1, i + 2):
            for col in range(j - 1, j + 2):
                #check if thing is not out of bounds
                if row >= 0 and row < h and col >= 0 and col < w:
                    if (row, col) not in self.moves_made:
                        cells.add((row, col))
        
        newKnowledge = Sentence(cells, count)
        self.knowledge.append(newKnowledge)

        s = []
        m = []
        
        for x in self.knowledge:
            for safes in x.known_safes():
                s.append(safes)
            for bombs in x.known_mines():
                m.append(bombs)
        
        for safe in s:
            self.mark_safe(safe)
        
        for mine in m:
            self.mark_mine(mine)
        

        
        #create new infrences
        count = 0
        cont = True
        while cont:
            for sents1 in self.knowledge:
                for sents2 in self.knowledge:
                    if sents1 is not sents2:
                        a = self.new_sentence(sents1, sents2)
                        b = self.new_sentence(sents2, sents1)

                        if a is not None and a not in self.knowledge:
                            self.knowledge.append(a)
                            count += 1
                        
                        if b is not None and b not in self.knowledge:
                            self.knowledge.append(b)
                            count += 1
            if count == 0:
                cont = False
            count = 0
        
        
    def new_sentence(self, sent1, sent2):
        """
        Creates a new sentence based on removing sent2 from sent1; returns none if not possible
        """
        cells = set()

        for cell in sent2.cells:
            if cell not in sent1.cells:
                return None

        for cell in sent1.cells:
            if cell not in sent2.cells:
                cells.add(cell)

        return Sentence(cells, sent1.count - sent2.count)

    

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for moves in self.safes:
            if moves not in self.moves_made:
                return moves
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        moves = []
        for row in range(0, self.height):
            for col in range(0, self.width):
                if (row, col) not in self.moves_made and (row, col) not in self.mines:
                    moves.append((row, col))
        
        if len(moves) > 0:
            return random.choice(moves)
        return None
