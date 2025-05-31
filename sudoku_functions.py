import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace


###################################################
## Define Class for Sudoku Puzzle
###################################################


class SudokuPuzzle:
    def __init__(self, n, q, square, flat, rows, cols, grids, unknowns):
        '''
        Inputs: 
        n ~ dimension of Sudoku puzzle; the full array is n^2 x n^2, with values from 1 to n^2
        q ~ number of qubits needed to store each digit; given by rounding up log2(n^2) to the nearest int
        square ~ square array of the puzzle (with digits 0 to n^2-1)
        flat ~ puzzle (with digits 0 to n^2-1) in a flat 1D array
        rows ~ contains the flat indices of the values in each row of the puzzle
        cols ~ contains the flat indices of the values in each column of the puzzle
        grids ~ contains the flat indices of the values in each grid of the puzzle
        unknowns ~ contains the flat indices of the unknown values in the puzzle
        '''
        self.n = n 
        self.q = q
        self.square = square
        self.flat = flat
        self.rows = rows
        self.cols = cols
        self.grids = grids
        self.unknowns = unknowns

    def row_checks(self, vis=True):
        '''
        Determines which rows contain unknown values (and need checked), and returns their indices 
        '''
        r = np.where(self.square<0)[0];
        chex = np.unique(r);
        vis and print("Need to check rows: ", chex + 1)
        return chex

    def col_checks(self, vis=True):
        '''
        Determines which columns contain unknown values (and need checked), and returns their indices
        '''
        c = np.where(self.square<0)[1];
        chex = np.unique(c);
        vis and print("Need to check columns: ", chex + 1)
        return chex

    def grid_checks(self, vis=True):
        '''
        Determines which grids contain unknown values (and need checked), and returns their indices
        '''
        chex = np.unique([np.where(self.grids == self.unknowns[j])[0] for j in range(len(self.unknowns))]);
        vis and print("Need to check grids: ", chex + 1);
        return chex

    def qubit_inds(self, u):
        '''
        Returns the indices of the qubits in the puzzle register corresponding to a particular unknown puzzle space
        '''
        assert self.flat[u]<0, "No qubits needed for known value"
        unknown_ind = np.where(self.unknowns==u)[0][0]
        return [self.q*unknown_ind+j for j in range(self.q)]

    def known_digits(self, u):
        '''
        Returns the binary representation of the value in the flat puzzle array at index u
        '''
        assert self.flat[u]>-1, "Value yet unkown"
        return f"{self.flat[u]:0{self.q}b}";


#######################################################################################
## Functions to process input puzzle from user, and output results from simulation
#######################################################################################


def process_input(puzzle_input, n):
    '''
    Takes in a square array `puzzle_input` of values between 1 and n^2, and converts it into a SudokuPuzzle class object
    '''
    assert puzzle_input.ndim==2 and puzzle_input.shape[0]==n**2 and puzzle_input.shape[1]==n**2, "Puzzle arrays must be n^2 x n^2"
    assert np.all((puzzle_input >= 0)) and np.all((puzzle_input <= n**2)), "Puzzle must have values 0 to n^2"
    
    puzzle_input -= 1; ## unknowns are -1, known values range from 0 to n^2-1
    puzzle_flat = puzzle_input.reshape(-1); ## puzzle reproduced as a 1D array

    ## coordinates of rows, columns, and grids in flat array
    row_coords = [ [n**2 * r + c for c in range(n**2)] for r in range(n**2)];
    col_coords = [ [n**2 * r + c for r in range(n**2)] for c in range(n**2)];
    grid_coords = [ [n**2 * (r+j) + (c+jj) for j in range(n) for jj in range(n)] for r in range(0,n**2,n) for c in range(0,n**2,n)];

    ## check the known values in each row, column, and grid don't have any errors to begin with
    for j in range(n**2):
        row = puzzle_flat[row_coords[j]]
        col = puzzle_flat[col_coords[j]]
        grid = puzzle_flat[grid_coords[j]]
        assert len(row[row>-1])==len(np.unique(row[row>-1])), f"Error in row {j}"
        assert len(col[col>-1])==len(np.unique(col[col>-1])), f"Error in col {j}"
        assert len(grid[grid>-1])==len(np.unique(grid[grid>-1])), f"Error in grid {j}"

    ## Determine number of qubits needed for each unknown value; given as number of binary digits needed to express values 0 to n^2-1
    q = np.ceil(np.log2(n**2)).astype(int);
    print("Qubits needed per unknown value: ", q)

    ## Determine total number of qubits needed for state of all unknown values 
    unknown_coords = np.where(puzzle_flat < 0)[0]
    Np = q*len(unknown_coords);
    print(len(unknown_coords), "unknown values at positions: ", unknown_coords);
    print("Puzzle qubits needed: ", Np)

    return SudokuPuzzle(n = n, q = q, square = puzzle_input, flat = puzzle_flat, rows = row_coords, cols = col_coords, grids = grid_coords, unknowns = unknown_coords)


def process_solutions(puzzle, qc):
    '''
    Takes an input puzzle and its corresponding Grover circuit; returns the histogram as the result of the circuit, 
    and converts the highest probability states into solutions into the n^2 x n^2 array format of the original puzzle
    '''
    ## get the final statevector of the circuit
    psi = Statevector(qc);
    ## trace out the state of everything but the puzzle qubits
    rho_p = partial_trace(psi, [j for j in range(puzzle.q*len(puzzle.unknowns), qc.num_qubits)]);

    ## get the probabilities for each state of the unknown puzzle values and plot a histogram
    p_states, probs = zip(*rho_p.probabilities_dict().items());
    p_labels = [int(p,2) for p in p_states]; ## convert binary states to decimal representations
    result_fig = plt.figure();
    result_ax = result_fig.add_subplot(111);
    result_ax.bar(p_labels, probs);
    result_ax.set_ylabel("Probability");
    result_ax.set_xlabel("Puzzle Qubit State");

    print(result_fig);

    ## Identify the states which are optimal (within 95% of the max)
    max_inds = [i for i, val in enumerate(probs) if val >= 0.95*max(probs)];
    max_states = [p_states[i] for i in max_inds];

    solns = [];
    ## for each optimal state (possible solution):
    for s in max_states:
        soln = np.copy(puzzle.flat); ## make a copy of the original puzzle
        for j in range(len(puzzle.unknowns)):
            ## grab the relevant digits from the binary state representation, convert to a decimal between 0 and n^2-1
            p = int( s[-(1+puzzle.q*j):-(puzzle.q+1+puzzle.q*j):-1], 2);
            soln[puzzle.unknowns[j]] = p ## plug this in to the appropriate spot in the solution

        soln = np.reshape(soln+1, (puzzle.n**2, puzzle.n**2)); ## reconstruct the square lattice solution
        solns.append(soln);
        print("\n Possible Solution: \n", soln )
    
    return solns, result_fig


###############################################################################################################
## Functions to generate circuits that will be applied in solving the sudoku
## The `oracle` function is the most involved, although much code is repeated as it loops over rows/cols/grids,
## and computes/uncomputes the different sets of ancillas
###############################################################################################################


def create_prep_circuit(puzzle, vis=False):
    '''
    For a given puzzle, determines the number of qubits needed to encode the unkonwn values, 
    and creates a circuit that initializes a register of qubits to an equal superposition of all basis states
    '''
    Np = puzzle.q*len(puzzle.unknowns); ## q qubits/value
    vis and print("Puzzle qubits: ", Np);
    puzzle_qr = QuantumRegister(size=Np, name = "p");

    prep_qc = QuantumCircuit(puzzle_qr, name = "State Prep");
    prep_qc.h(puzzle_qr); ## H^n |0>_n 

    vis and print(prep_qc.draw());
        
    return prep_qc.to_gate();


def create_map_to_ancilla(puzzle, u, vis=False):
    '''
    Creates the circuit that maps the known value at the flat index `u` onto a set of q ancillas 
    '''
    assert puzzle.flat[u]>-1, "Value must be known to map to ancillas"
    qr = AncillaRegister(size=puzzle.q, name = "a"); ## q qubits/value
    
    qc = QuantumCircuit(qr, name = f"Ancilla Map {puzzle.flat[u]}");
    
    digits = puzzle.known_digits(u);  
    ## flip the ancillas corresponding to the digits that are 1
    puzzle.flat[u]>0 and qc.x([qr[k] for k,bit in enumerate(digits) if bit=='1']);

    vis and print(qc.draw());

    return qc.to_gate();


def create_compare_digits(puzzle, vis=False):
    '''
    Creates the circuit to compare the digits of values stored in the registers u1_qr, u2_qr
    '''
    u1_qr = QuantumRegister(size=puzzle.q, name = "dA"); ## q qubits/value
    u2_qr = QuantumRegister(size=puzzle.q, name = "dB");
    check_qr = AncillaRegister(size=puzzle.q, name = "c"); ## store the value for each digit comparison in one qubit

    qc = QuantumCircuit(u1_qr, u2_qr, check_qr, name = "Digit Compare");
    
    for i in range(puzzle.q):
        ## if both digits are the same state, check_qr[i] state flips none/twice and is left unchanged;
        ## if digits are different, check_qr[i] is flipped
        qc.cx(u1_qr[i], check_qr[i]);
        qc.cx(u2_qr[i], check_qr[i]);

    vis and print(qc.draw());

    return qc.to_gate();


def create_or(puzzle, vis=False):
    '''
    Creates the circuit which computes logical "OR" on the qubits in `check_qr`, mapping the output to target_qr
    '''
    check_qr = AncillaRegister(size=puzzle.q, name = "c");
    target_qr = AncillaRegister(size=1, name = "r");

    qc = QuantumCircuit(check_qr, target_qr, name = "OR");

    ## if the bit is 1, it flips the target, then the bit is flipped to 0 to prevent any future gates from changing the target 
    ## if the bit is 0, it is flipped to 1 to allow future any future 1's to flip the target with MCX
    for i in range(puzzle.q-1):
        qc.mcx(check_qr[:(i+1)], target_qr[0]);
        qc.x(check_qr[i]);
    qc.mcx(check_qr, target_qr[0]);
    qc.x(check_qr[:-1]);

    vis and print(qc.draw());

    return qc.to_gate();


def create_diffuser(puzzle, vis=False):
    '''
    Creates the circuit for the Grover diffuser on the puzzle qubits of the unknown values
    '''
    qr = QuantumRegister(size=puzzle.q*len(puzzle.unknowns), name = "p"); ## q qubits/unknown value

    qc = QuantumCircuit(qr, name = "Diffuser");
    qc.h(qr);
    qc.x(qr);
    qc.mcp(np.pi, qr[:-1], qr[-1]);
    qc.x(qr);
    qc.h(qr);

    vis and print(qc.draw());

    return qc.to_gate()


def create_grover(puzzle, niter, vis=False):
    '''
    Assembles the full Grover circuit for the input puzzle, with the oracle and diffuser applied `niter` times.
    '''
    Np = puzzle.q*len(puzzle.unknowns);  ## q qubits/value
    p_qr = QuantumRegister(size=Np, name = "p");

    Na = puzzle.q;  ## q qubits/value for encoding known values
    a_qr = AncillaRegister(size=Na, name = "a");
    
    ## q qubits to compare digits, then (n^4-n^2)/2 inequalities in each row/col/grid
    Nc = int(puzzle.q + (puzzle.n**4-puzzle.n**2)/2);  
    c_qr = AncillaRegister(size=Nc, name = "c");

    ## one qubit for each row/col/grid checked
    Nr = len(puzzle.row_checks(vis=False)) + len(puzzle.col_checks(vis=False)) + len(puzzle.grid_checks(vis=False)); 
    r_qr = AncillaRegister(size=Nr, name = "r");

    qc = QuantumCircuit(p_qr, a_qr, c_qr, r_qr, name = "Grover");
    
    # initialize needed circuits
    prep = create_prep_circuit(puzzle);  
    oracle = create_oracle_circuit(puzzle);
    diffuser = create_diffuser(puzzle);

    ## Prepare initial state
    qc.compose(prep, inplace=True);

    for j in range(niter):
        ## apply oracle/diffuser to phase invert solutions, and invert about the mean to amplify the solution probability
        qc.compose(oracle, inplace=True);
        qc.compose(diffuser, inplace=True);

    vis and print(qc.draw())

    return qc


def create_oracle_circuit(puzzle, vis=False):
    '''
    Creates the Grover oracle for the puzzle, which identifies the state of valid solutions and inverts the phase of those states,
    allowing the diffuser to amplify the weight of the solutions in the overall state.
    '''
    ## determine which rows/cols/grids to check
    row_checks = puzzle.row_checks(vis); 
    col_checks = puzzle.col_checks(vis);
    grid_checks = puzzle.grid_checks(vis);
    q = puzzle.q;
    
    Np = q*len(puzzle.unknowns);  ## q qubits/value
    puzzle_qr = QuantumRegister(size=Np, name = "p");
    vis and print("Puzzle qubits: ", Np)

    Na = q;  ## q qubits/value for encoding known values
    ancilla_qr = AncillaRegister(size=Na, name = "a");
    vis and print("Known value ancilla qubits: ", Na)
    
    ## q qubits to compare digits, then (n^4-n^2)/2 inequalities in each row/col/grid
    Nc = int(q + (puzzle.n**4-puzzle.n**2)/2);
    check_qr = AncillaRegister(size=Nc, name = "c");
    vis and print("Inequality check qubits : ", q ," + ", Nc-q)

    ## one qubit for each row/col/grid checked
    Nr = len(row_checks) + len(col_checks) + len(grid_checks);
    result_qr = AncillaRegister(size=Nr, name = "r");
    vis and print("Row/Col/Grid check qubits: ", Nr)

    N_tot = Np+Na+Nc+Nr;  ## total number of qubits
    print("\n Total number of qubits: ", N_tot)

    oracle_qc = QuantumCircuit(puzzle_qr, ancilla_qr, check_qr, result_qr, name = "Oracle");

    ## Initialize subroutines needed for comparing puzzle values
    compare_digits = create_compare_digits(puzzle);
    OR_gate = create_or(puzzle);

    rs = 0; ## iterate over total number of checks; increment after check complete
    
##############################################################################################################################
    
    for r in row_checks:
        row = puzzle.rows[r];    
        
        for t, (u1,u2) in enumerate(list(combinations(row,2))):
            ## Determine values at indices in the row/col/grid
            val1 = puzzle.flat[u1];
            val2 = puzzle.flat[u2];
            
            ## Four cases: depending on if val1, val2 are known or not
            if val1>-1 and val2>-1:  ## If both values are known,
                oracle_qc.x(check_qr[t]); ## Just flip the corresponding check bit, we already verified this
                    
            elif val1<0 and val2<0: ## If both values are unknown,
                ## find the corresponding qubits for these unknowns,
                a = puzzle.qubit_inds(u1);
                b = puzzle.qubit_inds(u2);
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                        
            elif val1>-1 and val2<0:  ## If the first value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u1);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u2);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
                        
            elif val1<0 and val2>-1:  ## If the second value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u2);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u1);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
            
        #####################################################################
        ## if all the inequalities are satisfied, flip the corresponding result bit,
        oracle_qc.mcx(check_qr[:(Nc-q)], result_qr[rs]);
        ## and increment the result counter
        rs+=1; 
        #####################################################################

        ## Always uncompute check_qr[:(Nc-q)];
        for t, (u1,u2) in enumerate(list(combinations(row,2))):
            ## Determine values at indices in the row/col/grid
            val1 = puzzle.flat[u1];
            val2 = puzzle.flat[u2];
            
            ## Four cases: depending on if val1, val2 are known or not
            if val1>-1 and val2>-1:  ## If both values are known,
                oracle_qc.x(check_qr[t]); ## Just flip the corresponding check bit, we already verified this
                    
            elif val1<0 and val2<0: ## If both values are unknown,
                ## find the corresponding qubits for these unknowns,
                a = puzzle.qubit_inds(u1);
                b = puzzle.qubit_inds(u2);
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                        
            elif val1>-1 and val2<0:  ## If the first value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u1);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u2);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
                        
            elif val1<0 and val2>-1:  ## If the second value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u2);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u1);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
    
##############################################################################################################################
    
    for c in col_checks:
        col = puzzle.cols[c];
        
        for t, (u1,u2) in enumerate(list(combinations(col,2))):
            ## Determine values at indices in the row/col/grid
            val1 = puzzle.flat[u1];
            val2 = puzzle.flat[u2];
            
            ## Four cases: depending on if val1, val2 are known or not
            if val1>-1 and val2>-1:  ## If both values are known,
                oracle_qc.x(check_qr[t]); ## Just flip the corresponding check bit, we already verified this
                    
            elif val1<0 and val2<0: ## If both values are unknown,
                ## find the corresponding qubits for these unknowns,
                a = puzzle.qubit_inds(u1);
                b = puzzle.qubit_inds(u2);
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                        
            elif val1>-1 and val2<0:  ## If the first value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u1);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u2);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
                        
            elif val1<0 and val2>-1:  ## If the second value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u2);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u1);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

        #####################################################################
        ## if all the inequalities are satisfied, flip the corresponding result bit,
        oracle_qc.mcx(check_qr[:(Nc-q)], result_qr[rs]);
        ## and increment the result counter
        rs+=1; 
        #####################################################################
        
        ## Always uncompute check_qr[:(Nc-q)];
        for t, (u1,u2) in enumerate(list(combinations(col,2))):
            ## Determine values at indices in the row/col/grid
            val1 = puzzle.flat[u1];
            val2 = puzzle.flat[u2];
            
            ## Four cases: depending on if val1, val2 are known or not
            if val1>-1 and val2>-1:  ## If both values are known,
                oracle_qc.x(check_qr[t]); ## Just flip the corresponding check bit, we already verified this
                    
            elif val1<0 and val2<0: ## If both values are unknown,
                ## find the corresponding qubits for these unknowns,
                a = puzzle.qubit_inds(u1);
                b = puzzle.qubit_inds(u2);
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                        
            elif val1>-1 and val2<0:  ## If the first value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u1);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u2);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
                        
            elif val1<0 and val2>-1:  ## If the second value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u2);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u1);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);      

##############################################################################################################################
    
    for g in grid_checks:
        grid = puzzle.grids[g];
        
        for t, (u1,u2) in enumerate(list(combinations(grid,2))):
            ## Determine values at indices in the row/col/grid
            val1 = puzzle.flat[u1];
            val2 = puzzle.flat[u2];
            
            ## Four cases: depending on if val1, val2 are known or not
            if val1>-1 and val2>-1:  ## If both values are known,
                oracle_qc.x(check_qr[t]); ## Just flip the corresponding check bit, we already verified this
                    
            elif val1<0 and val2<0: ## If both values are unknown,
                ## find the corresponding qubits for these unknowns,
                a = puzzle.qubit_inds(u1);
                b = puzzle.qubit_inds(u2);
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                        
            elif val1>-1 and val2<0:  ## If the first value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u1);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u2);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
                        
            elif val1<0 and val2>-1:  ## If the second value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u2);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u1);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);   
            
        #####################################################################
        ## if all the inequalities are satisfied, flip the corresponding result bit,
        oracle_qc.mcx(check_qr[:(Nc-q)], result_qr[rs]);
        ## and increment the result counter
        rs+=1; 
        #####################################################################
        
        ## Always uncompute check_qr[:(Nc-q)];
        for t, (u1,u2) in enumerate(list(combinations(grid,2))):
        ## Determine values at indices in the row/col/grid
            val1 = puzzle.flat[u1];
            val2 = puzzle.flat[u2];
            
            ## Four cases: depending on if val1, val2 are known or not
            if val1>-1 and val2>-1:  ## If both values are known,
                oracle_qc.x(check_qr[t]); ## Just flip the corresponding check bit, we already verified this
                    
            elif val1<0 and val2<0: ## If both values are unknown,
                ## find the corresponding qubits for these unknowns,
                a = puzzle.qubit_inds(u1);
                b = puzzle.qubit_inds(u2);
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*[puzzle_qr[i] for i in a], *[puzzle_qr[i] for i in b],*check_qr[-q:]), inplace=True);
                        
            elif val1>-1 and val2<0:  ## If the first value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u1);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u2);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);
                        
            elif val1<0 and val2>-1:  ## If the second value is known, 
                ## map the known value to ancillas,
                map_to_ancilla = create_map_to_ancilla(puzzle, u2);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);

                ## find the qubits for the unknown value,
                b = puzzle.qubit_inds(u1);
    
                ## and compare their digits; if at least one digit is different, then the inequality is satisfied and check_qr[t] is flipped.
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(OR_gate, qubits=(*check_qr[-q:], check_qr[t]), inplace=True);
                
                ## Always uncompute the ancillas
                oracle_qc.compose(compare_digits, qubits=(*ancilla_qr, *[puzzle_qr[i] for i in b], *check_qr[-q:]), inplace=True);
                oracle_qc.compose(map_to_ancilla, qubits=ancilla_qr, inplace=True);    

    ## After checking every inequality needed, do MCZ on result bits
    oracle_qc.mcp(np.pi, result_qr[:-1], result_qr[-1]);
    vis and print(oracle_qc.draw())
    
    return oracle_qc.to_gate()

