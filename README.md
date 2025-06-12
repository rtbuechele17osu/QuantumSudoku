# QuantumSudoku
Solving Sudoku puzzles with Grover's Search Algorithm - a mini-project for the [Erdos Quantum Computing Boot Camp](https://www.erdosinstitute.org/programs/summer-2025/quantum-computing-boot-camp).

In the "Mini Sudoku" portion of the notebook, I demonstrate how to solve a simplified Sudoku puzzle of the form 

$$
\left[\begin{array}{c|c}
a & b \\
\hline
c & d \\
\end{array}\right]; \{a,b,c,d\} \in \{0,1\}.
$$

by encoding the puzzle onto the quantum state of 4 qubits and using an implementation of Grover's search algorithm to identify which states/configurations are valid solutions to the puzzle.

In general, the sudoku puzzle is a $n^2 \times n^2$ grid of values $1$ through $n^2$, subdivided into an $n\times n$ array of $n\times n$ subgrids. Each value is used exactly once in each row, column, and subgrid. In the "Full Sudoku" portion of the notebook, a user can enter the dimension `n` and known values of a puzzle in the cell as shown: 
```
## Enter n for dimension of puzzle
n = 2; 

## Enter values between 1 and n^2; leave unknown values as 0s
puzzle_input = np.array([[ 1 , 2 , 4 , 3 ],
                         [ 3 , 0 , 1 , 2 ],
                         [ 2 , 1 , 0 , 4 ],
                         [ 0 , 3 , 2 , 1]]);  

puzzle = process_input(puzzle_input, n);
```

From here, the provided functions map the configurations of the unknown values onto a quantum state and build the Grover circuit which identifies valid solutions to the given puzzle. Finally, it outputs the statevector of this Circuit and converts the solution state back into an array of values, showing the solved Sudoku:

```
 Possible Solution: 
 [[1 2 4 3]
 [3 4 1 2]
 [2 1 3 4]
 [4 3 2 1]]
```

Necessary functions with comments included in sudoku_functions.py, user input and documentation/description included in the Jupyter notebook; requires matplotlib and qiskit.
