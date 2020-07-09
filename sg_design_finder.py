# 7/8/2020
# rewrite of simialr code form scratch

import pynauty
import itertools
import numpy as np
import copy
from solve_exact_cover import *
import dill

"""
PartialDesign class for recording point-line designs
It is a partial design because not every two points has a line passing through 
it, only some do. 
"""
class PartialDesign:
    # Initialize partial design with num_points points, and some initial set 
    # of lines. 
    # - num_points: number of points in the design
    # - lines: a list of L lists, each of which is a line to be added.
    #   Lines are represented as a list of numbers 0 <= k < num_points, 
    #   the points on that line.
    def __init__(self, num_points, lines):
        self.lines = []
        self.num_points = num_points
        self.has_line = np.zeros((num_points, num_points), dtype=np.bool)

        for l in lines:
            self.add_line(l)

    # l = [x_0, ..., x_k], the points on a line to be added. Adds that lines to
    # self.lines, and updates the "has_line" array as well.
    def add_line(self, l):
        l = tuple(sorted(l))
        if l in self.lines:
            return

        self.lines.append(l)
        for v1 in l:
            for v2 in l:
                if v1 == v2:
                    continue
                # if this was already true, then line is illegal
                assert(self.has_line[v1, v2] == False)

                # there is now a line passing through v1, v2
                self.has_line[v1, v2] = True

    # remove a line from self.lines if it exists, and update "has_line" array
    def remove_line(self, l):
        l = tuple(sorted(l))
        if l not in self.lines:
            return

        self.lines.remove(l)
        for v1 in l:
            for v2 in l:
                if v1 == v2:
                    continue
                self.has_line[v1, v2] = False

    # check if a line passes through two points already lying on a line
    def can_add(self, l):
        for v1 in l:
            for v2 in l:
                if v1 == v2:
                    continue
                if self.has_line[v1, v2]:
                    return False
        return True

"""
Make a bipartite incidence graph for a point-line design
- pd - partial design object representing point-line design
- Returns a pynauty bipartite graph object. 
  The points will correspond to one set of vertices, and lines correspond to the
  others. The edges of the bipartite graph record point-line incidences.
"""
def make_bipartite_for_design(pd):
    # we have num points + num lines vertices
    # first are the points, then then lines
    # the points have color 0 and lines have color 1
    num_points = pd.num_points
    num_lines = len(pd.lines)
    num_vert = num_points + num_lines

    # different colors for lines and for points of each pencil type
    lineset = set(range(num_points,num_vert))
    coloring = [set(range(num_points)), set(range(num_points,num_vert))]

    adj_dict = {}
    for i in range(num_vert):
        adj_dict[i] = []

    # li is the line index, 0, ..., L-1
    # line is the set of vertices in the line, numbers 0 <= x < N
    for li, line in enumerate(pd.lines):
        line_ind = num_points + li # line index in graph is N + li
        for p in line:
            # add an edge in G between line_ind and p
            adj_dict[line_ind].append(p)
            adj_dict[p].append(line_ind)

    g = pynauty.Graph(number_of_vertices = num_vert, \
        directed = False, \
        adjacency_dict = adj_dict, \
        vertex_coloring = coloring)

    return g

def make_bipartite_for_design_linelist(num_points, line_list):
    # we have num points + num lines vertices
    # first are the points, then then lines
    # the points have color 0 and lines have color 1
    num_lines = len(line_list)
    num_vert = num_points + num_lines
    # different colors for lines and for points of each pencil type
    lineset = set(range(num_points, num_vert))
    coloring = [set(range(num_points)), set(range(num_points,num_vert))]
    
    adj_dict = {}
    for i in range(num_vert):
        adj_dict[i] = []
    for li, line in enumerate(line_list):
        line_ind = num_points + li
        for p in line:
            adj_dict[line_ind].append(p)
            adj_dict[p].append(line_ind)

    g = pynauty.Graph(number_of_vertices = num_vert, \
        directed = False, \
        adjacency_dict = adj_dict, \
        vertex_coloring = coloring)

    return g

# creates a unique identifier has for a partial design
# used for isomorphism testing
def make_identifier_hash(pd):
    return pynauty.certificate(make_bipartite_for_design(pd))


def make_identifier_hash_linelist(npoints, line_list):
    return pynauty.certificate(make_bipartite_for_design_linelist(npoints, line_list))


"""
make a list containing all possible lines in n vertices, up to length maxlen
returns two objects containing this data:
 - a dictionary with keys being a length 3 <= L <= maxlen, and values being 
   all the lines with precisely that length
 - a list of lists, with the k,l entry being all the lines of length l through 
   the point k. 
"""
def make_all_lines(n, maxlen = None):
    if maxlen == None:
        maxlen = n

    all_lines_with_len = {}

    # initialize list of all lines for each possible length
    for i in range(3, maxlen+1):
        all_lines_with_len[i] = list(itertools.combinations(list(range(n)), i))

    lines_through_each_with_len = [{} for i in range(n)]

    # intiialize empty array for lines of each length through point
    for i in range(n):
        for j in range(3, maxlen+1):
            lines_through_each_with_len[i][j] = []

    # add relevant lines to entries in array
    for line_len in range(3, maxlen+1):
        for line in all_lines_with_len[line_len]:
            for p in line:
                lines_through_each_with_len[p][line_len].append(line)

    return all_lines_with_len, lines_through_each_with_len

"""
Rewriting of function "find_all_seeds2" from prior version
Finds all nonisomorphic ways to add all lines through some initial set of points
 - npoints: number of points on which to find seeds
 - maxlen: maximum line length to consider
 - initial_len: length of line to find seeds from. 
 - All lines will have at least this length
"""
def find_all_seeds(npoints, initial_len, pt_up_to, verbose=True):
    min_line_len = initial_len
    maxlen = npoints//2 + 1

    if verbose: 
        print(("Finding all seeds: npoints = {}, maxlen = {}, " + 
            "initial_len = {}").format(npoints, maxlen, initial_len))

    if verbose: print("Initializing lines")
    all_lines_with_len, lines_through_each_with_len = make_all_lines(npoints)
    initial_line = range(initial_len)
    if verbose: print("Finished intializing")

    

    # base partial design, consisting of only the base line
    base_pd = PartialDesign(npoints, [initial_line]) 
    base_h = make_identifier_hash(base_pd)

    # five values: partial design, hash, point on, line length on, option on
    # each stack entry is a breadth first search branch to explore
    look_at_stack = [{"pd" : base_pd, 
                     "hash" : base_h,
                     "pt_on" : 0, 
                     "length_on": maxlen,
                     "option_on": 0}]

    completed_stack = [] # completed designs, to return at end
    known_hashes = set([(base_h)]) # design hashes that have appeared so far

    run = 0 # run on 

    # while there are search brancehs to explore, pop them and keep exploring
    while len(look_at_stack) > 0:
        search_entry = look_at_stack.pop(0)
        pd = search_entry["pd"]
        pd_h = search_entry["hash"]
        pt_on = search_entry["pt_on"]
        line_len_on = search_entry["length_on"]
        option_on = search_entry["option_on"]

        nlines = len(pd.lines)

        # logging info
        if verbose and run >= 0 and run % 10000 == 0:
            print("\nOn run:", run, ", in queue:", len(look_at_stack), 
                ",  solutions found:", len(completed_stack))
            print("num lines =", nlines, ", lines:")
            for l in pd.lines:
                print(l)

        # if pt_on is after the initial line, add design to stack and 
        # terminate this branch
        if pt_on == pt_up_to:
            completed_stack.append(pd)
            continue

        # number of lines passing through each point in configuration
        pts_connected = np.sum(pd.has_line, axis = 1)

        if pts_connected[pt_on] == pd.num_points - 1: # point all filled up
            look_at_stack.append({
                "pd": pd,
                "hash": pd_h, 
                "pt_on": pt_on+1,
                "length_on": maxlen,
                "option_on" : 0})
            continue

        # try adding all lines through pt_on
        candidate_lines = lines_through_each_with_len[pt_on][line_len_on]

        # exhausted options, decrease line length
        if option_on == len(candidate_lines): 
            if line_len_on > min_line_len:
                look_at_stack.append({
                    "pd": pd, 
                    "hash": pd_h, 
                    "pt_on": pt_on, 
                    "length_on": line_len_on-1, 
                    "option_on": 0})
            continue

        chosen_line = candidate_lines[option_on]

        # if we can add the chosen line, do so
        if pd.can_add(chosen_line):
            # first push command to remove line and proceed without this line
            pd.add_line(chosen_line)
            h = make_identifier_hash(pd)
            pd.remove_line(chosen_line)

            if h not in known_hashes:
                known_hashes.add(h)
                new_pd = copy.deepcopy(pd)
                new_pd.add_line(chosen_line)
                branch_with_newline = {
                    "pd": new_pd, 
                    "hash": h, 
                    "pt_on": pt_on, 
                    "length_on": line_len_on, 
                    "option_on": option_on+1}
                look_at_stack.append(branch_with_newline)

            # then push command to add this line and proceed with it
            branch_no_newline = {
                "pd": pd, 
                "hash": pd_h, 
                "pt_on": pt_on, 
                "length_on": line_len_on, 
                "option_on": option_on+1}

            look_at_stack.insert(0, branch_no_newline) # insert at front

        # if we can't add the chosen line, go to next option
        else:
            # neither add nor remove lime
            command = {
                "pd": pd, 
                "hash": pd_h, 
                "pt_on": pt_on, 
                "length_on": line_len_on, 
                "option_on": option_on+1}

            look_at_stack.insert(0, command) # insert at front

        run += 1

    if verbose: print("Finished with:", len(completed_stack), " solutions!")
    return completed_stack

# given a partial design, find all ways to add lines such that a specified point
# contains a complete pencil
def enumerate_saturations(psol, point_saturate, known_hashes = set([])):
    n  = psol.num_points
    maxlen = n//2 + 1

    # make relevant lines 
    # note: if necessary, we can get a speed up by moving this to 
    # `enumerate_full_solutions' function)
    all_lines_with_len, lines_through_each_with_len = make_all_lines(n)

    pts_to_cover = [i for i in range(n) if \
    i != point_saturate and psol.has_line[point_saturate, i] == False]

    underlying_set = set(range(len(pts_to_cover)))
    valid_lines = []
    for l in range(3, maxlen+1):
        choices = lines_through_each_with_len[point_saturate][l]
        for c in choices:
            if psol.can_add(c):
                valid_lines.append(c)  

    # Y is the set of covering sets
    Y = {}
    for i,l in enumerate(valid_lines):
        Y[i] = []
        for j in l:
            if j == point_saturate:
                continue
            else:
                pind = pts_to_cover.index(j)
                Y[i].append(pind)

    # X1 and Y1 are disctionaries on which we an run algorithm X
    X1, Y1 = make_inputs(underlying_set, Y)
    all_saturations = []

    for s in solve(X1, Y1):
        add_lines = [valid_lines[j] for j in s]
        h = make_identifier_hash_linelist(n, psol.lines + add_lines)
        if h not in known_hashes:
            known_hashes.add(h)
            new_pd = copy.deepcopy(psol)
            for l in add_lines:
                new_pd.add_line(l)
            # print("num total full sol:", len(known_hashes_full))
            all_saturations.append(new_pd)
    return all_saturations

def all_full_completions(psol, known_hashes_full, minlen = 3):
    n  = psol.num_points
    maxlen = n//2 + 1

    # make relevant lines 
    # note: if necessary, we can get a speed up by moving this to 
    # `enumerate_full_solutions' function)
    all_lines_with_len, lines_through_each_with_len = make_all_lines(n)

    all_pairs = []
    # add all pairs that don't already have a line
    for i in range(2,n):
        for j in range(i+1,n):
            if not psol.has_line[i,j]:
                all_pairs.append((i,j))
    underlying_set = set(range(len(all_pairs)))

    valid_lines = []
    for l in range(minlen, maxlen+1):
        choices = all_lines_with_len[l]
        for c in choices:
            if psol.can_add(c):
                valid_lines.append(c)  

    # Y is the set of covering sets
    Y = {}
    for i,l in enumerate(valid_lines):
        Y[i] = []
        for i1 in l:
            for i2 in l:
                if i1 >= i2:
                    continue
                if i1 <= 1 or i2 <= 1:
                    continue
                pind = all_pairs.index((i1,i2))
                # print(i, l, i1, i2, pind)
                Y[i].append(pind)

    # X1 and Y1 are disctionaries on which we an run algorithm X
    X1, Y1 = make_inputs(underlying_set, Y)
    all_completions = []
    for s in solve(X1, Y1):
        add_lines = [valid_lines[j] for j in s]
        h = make_identifier_hash_linelist(n, psol.lines + add_lines)
        if h not in known_hashes_full:
            known_hashes_full.add(h)
            new_pd = copy.deepcopy(psol)
            for l in add_lines:
                new_pd.add_line(l)
            # print("num total full sol:", len(known_hashes_full))
            all_completions.append(new_pd)
    return all_completions


# Enumerate all Sylvester-Gallai designs on npoints points with minimum line 
# length 3. 
def enumerate_full_solutions_min3(npoints):
    print(("Finding Sylvester-Gallai designs on {} points " + 
        "with min length three").format(npoints))
    maxlen = npoints // 2 + 1

    # first find seeds on 2 vertices
    all_seeds = find_all_seeds(npoints, 3, 2)

    # then add seeds on the 3rd vertex also on the initial line on 3 vertices
    all_line_completions = []
    known_hashes_first_line = set([])
    for i,s in enumerate(all_seeds):
        print("Completing first line:", i, '/', len(all_seeds), ':',
            len(known_hashes_first_line))
        all_line_completions += enumerate_saturations(s, 2, 
            known_hashes_first_line)

    print("Found first line completions:", len(all_line_completions))

    # finally find all completions our set of seeds
    all_sg_completions = []
    known_full_hashes = set([])
    for i,s in enumerate(all_line_completions):
        print("Completing full design:", i, '/', len(all_line_completions), ':',
            len(known_full_hashes))
        all_sg_completions += all_full_completions(s, known_full_hashes)

    print("Found full designs:", len(all_sg_completions))
    return all_sg_completions

# Enumerate all Sylvester-Gallai designs on npoints points with minimum line 
# length greater than or equal to 4. 
def enumerate_full_solutions_min4(npoints):
    print(("Finding Sylvester-Gallai designs on {} points " + 
            "with min length four or more").format(npoints))    
    maxlen = npoints // 2 + 1
    all_seeds = []

    # first find all seeds, saturating an initial lin with minlen points
    for minlen in range(4, maxlen):
        all_seeds += find_all_seeds(npoints, minlen, minlen)

    all_sg_completions = []
    known_full_hashes = set([])

    # then find all completions of the seeds, making sure to use lines 
    # of length at least four
    for i,s in enumerate(all_seeds):
        print("Completing full design:", i, '/', len(all_seeds), ':',
            len(known_full_hashes))
        all_sg_completions += all_full_completions(s, known_full_hashes, 4)
        

    print("Found full designs with min length at least four:", 
        len(all_sg_completions))
    return all_sg_completions

def enumerate_all_sg_designs(npoints):
    designs_big = enumerate_full_solutions_min4(npoints)
    designs_three = enumerate_full_solutions_min3(npoints)
    return designs_big + designs_three


# make all solutions for 7 through 16 points, and save them all to files
for npoints in range(7, 17):
    print("------------------\nFINDING ALL DESIGNS: {}\n------------".format(npoints))
    my_solutions = enumerate_all_sg_designs(npoints)

    with open("saved_classification/all_unique_sg_{}.txt".format(npoints), "w") as dataf:
        dataf.write("All unique sylvester gallai designs on {} points\n".format(npoints))
        for i, pd in enumerate(my_solutions):
            dataf.write("Sylvester Gallai design " + str(i+1) + "/" + str(len(my_solutions)) + "\n")
            for l in pd.lines:
                dataf.write(str(l))
                dataf.write("\n")
            dataf.write("\n")

    all_trips = [pd.lines for pd in my_solutions]
    with open("saved_classification/all_unique_sg_{}.dill".format(npoints), "wb") as dillf:
        dill.dump(all_trips, dillf)
