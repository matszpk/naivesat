## NaiveSAT

The naive SAT solver and other solver for problems. These programs uses the Gate Project
libraries including gatenative that provides native simulation.
These programs solve problem using a naive brute force algorithm.

Packages provides few utilities:

### naivesat

This program try to solve SAT problem for Gate circuits. A circuit must have only 1 output.
Problem is finding that combinations of circuit inputs that circuit returns 1 at its output.

### naiveqsat

This program try to solve QSAT problem for Gate circuits. A circuit must have only 1 output
and must have provided quantifiers.
Problem QSAT is similar to SAT problem, however it solve that problem:

Q0 X0 Q1 X2 ... Qn Xn circuit(X0,....,Xn)==1

Qx - quantifier for x'th circuit input: 'all' or 'exists'.
Xx - circuit input.

### naivesatostop

This program try to solve SATOSTOP problem. The SATOSTOP problem is combination of
the halting problem for bounded state (in this case circuits) and SAT problem.
A circuit must have circuit output greater by 1 than circuit input. Last circuit output is
STOP bit.

In this problem a circuit will be executed multiple times. In first time circuit
input state is 0. Next time, circuit fetched output state from previous simulation.
An execution stops if circuit returns 1 in STOP bit.

UNKNOWNS is number of unknown circuit inputs. Problem is finding that combination of
unknowns that circuit exeuction stops.

### greedysatostop

Similar to naivesatostop but can uses storage to store intermediate data - table of states.
The program stores in file big table of states, load to memory its partitions to
make calculations and store back to file.
