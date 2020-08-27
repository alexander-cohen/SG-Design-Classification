import sys
# from sage.all import *
import dill
import numpy as np
import random
import itertools
from sympy import *
from sg_design_finder import PartialDesign

def load_configs(n):
    fn = "saved_classification/all_unique_sg_" + str(n) + ".dill"
    with open(fn, "rb") as dillf:
        all_solutions_lines = dill.load(dillf)
        all_designs = [PartialDesign(n, lines) for lines in all_solutions_lines]

    return all_designs

# load all configs
all_configs = {}
for n in range(7, 17):
    all_configs[n] = load_configs(n)

# compute the set of points determined by the position of "choices"
# think of it as a game on a graph: we color some points red, then at each step, 
# color a white vertex red if its adjacent to two red vertices 
# eg point on two lines or line on two points
def forced_set(pd, initials):

    # adjacency matrix for bipartite graph
    # the i, j column is 1 if that point & line are connected, 0 otherwise
    adj_mat = np.zeros((pd.num_points, len(pd.lines)), dtype=np.int) 
    for line_num,l in enumerate(pd.lines):
        for pt_num in l:
            adj_mat[pt_num, line_num] = 1

    # 0-1 vector recording which vertices & lines are hit
    hit_points = np.zeros(pd.num_points, dtype=np.int)
    hit_lines = np.zeros(len(pd.lines), dtype=np.int)
    for pt_num in initials:
        hit_points[pt_num] = 1

    pt_order = []
    lines_defining_pt = {}
    pts_defining_line = {}

    for i in initials:
        pt_order.append(i)
        lines_defining_pt[i] = []

    

    # we compute how many hit lines are adjacent by adj_mat * hit_lines
    # we compute how many hit vertices are adjacent by hit_lines * adj_mat
    # we set those with > 1 hit to true

    did_add = True
    while did_add:
        did_add = False
        num_hit_adj_pts = np.matmul(adj_mat, hit_lines)
        num_hit_adj_lines = np.matmul(np.transpose(adj_mat), hit_points)

        for i in range(len(pd.lines)):
            if num_hit_adj_lines[i] > 1 and hit_lines[i] == 0:
                did_add = True
                hit_lines[i] = 1

                dotted = np.transpose(adj_mat)[i,:] * hit_points
                nz = np.nonzero(dotted)[0]
                pts_defining_line[i] = [nz[0], nz[1]]

        for i in range(pd.num_points):
            if num_hit_adj_pts[i] > 1 and hit_points[i] == 0:
                did_add = True
                hit_points[i] = 1
                pt_order.append(i)

                dotted = adj_mat[i,:] * hit_lines
                nz = np.nonzero(dotted)[0]
                lines_defining_pt[i] = [nz[0], nz[1]]


    # each entry has 5 points: a point to add, 
    # and two tuples of points defining lines the first point lies on
    force_order = []
    for p in pt_order:
        lines_def = lines_defining_pt[p]
        pts_defining_lines = [pts_defining_line[l] for l in lines_def]
        force_order.append((p, pts_defining_lines))

    return [i for i in range(pd.num_points) if hit_points[i]], force_order



# test if a subset of points is a forcing fixture
# choices are point indices
def is_forcing_fixture(pd, initials):
    fs, force_order = forced_set(pd, initials)
    return len(fs) == pd.num_points, force_order

def has_three_collinear(pd, initial):
    for l in pd.lines:
        if len(set(l).intersection(set(initial))) >= 3:
            return True
    else:
        return False

# try to find a set of points which determine the position of the other points
def find_forcing_fixture(pd):
    for fs_size in range(3, pd.num_points):
        # print("Trying to find fixture of size:", fs_size)
        for initial in itertools.combinations(range(pd.num_points), fs_size):
            if has_three_collinear(pd, initial): # fixture set must have no three collinear
                continue
            is_forcing, force_order = is_forcing_fixture(pd, initial)
            if is_forcing:
                print("Found fixture of size:", fs_size)
                # print("Force order:")
                # for dat in force_order:
                #     print(dat)
                return initial, force_order

    return False, False

def is_constant(expr):
    return len(expr.free_symbols) == 0

def make_pt_coords(pd, force_order):
    x0, y0 = symbols('x0 y0')
    first_four_vecs = [Matrix([0,0,1]),
            Matrix([0,1,0]),
            Matrix([1,0,0]),
            Matrix([1,1,1]),
            Matrix([x0,y0,1])]

    pt_coords = [None for i in range(pd.num_points)]
    initial_on = 0
    for pt, fixt in force_order:
        if fixt == []:
            pt_coords[pt] = first_four_vecs[initial_on]
            initial_on += 1

        else:
            p1 = fixt[0][0]
            p2 = fixt[0][1]
            q1 = fixt[1][0]
            q2 = fixt[1][1]
            # point lies on intersection of line through (p1, p2) and (q1, q2)

            # make lines through the defining points, and intersect
            l1 = pt_coords[p1].cross(pt_coords[p2])
            l2 = pt_coords[q1].cross(pt_coords[q2])

            pc = l1.cross(l2)
            pt_coords[pt] = expand(pc)

    return pt_coords

def make_eqs(pd, pt_coords):
    # comes from colinear triples
    zero_polys = []

    all_threesets = {}
    for trip in itertools.combinations(range(pd.num_points), 3):
        all_threesets[trip] = False

    # the triples appearing from lines should be collinear
    # the rest should not be
    for line in pd.lines:
        for trip in itertools.combinations(line, 3):
            all_threesets[trip] = True

    for trip in all_threesets:
        M = Matrix([list(pt_coords[trip[0]]), 
                    list(pt_coords[trip[1]]), 
                    list(pt_coords[trip[2]])])
        eq = M.det()
        eq = expand(eq)
        if all_threesets[trip]: # should be collinear
            if eq != 0 and is_constant(eq):
                print("Found constant polynomial:", eq)
                return False 
            elif eq != 0:
                zero_polys.append(eq)
            
        elif eq == 0:
            print("Found zero poly that should have been nonzero:", eq)
            return False

    return zero_polys

def resolve_eqs(eqs, method = "grlex"):
    x0, y0 = symbols('x0 y0')
    z = symbols('z')
    extra_eq = 1 - z * x0*y0*(1-x0)*(1-y0)*(x0-y0)
    grob_e = eqs + [extra_eq]
    G = groebner(grob_e, x0, y0, z, order=method)

    return G

def possibly_embeddable(pd):
    init, fo = find_forcing_fixture(pd)
    if init == False:
        print("Could not find fixture of size <= 5 with no three collinear, could be embeddable")
        return True
    if len(init) > 5:
        print("Could not find fixture of size 5, could be embeddable")
        return True

    pt_coords = make_pt_coords(pd, fo)
    zero_eqs = make_eqs(pd, pt_coords)
    if zero_eqs == False:
        print("Became clear making equations that design is not embeddable")
        return False

    G = resolve_eqs(zero_eqs)
    if G == [1]:
        print("Found 1 in ideal, not embeddable")
        return False
    else:
        print("Did not find 1 in ideal, could be embeddable")
        return True


# all but one of the configurations have a fixture of size <= 5
# this is 67/119 for 15 pts
# we have an argument for this case: the four points 3,4,5,6
# have 3 pairs of two lines passing through them
# this is impossible

possibly_embeddable_pds = []
for num_pts in range(7, 17):
    print("\n\nOn designs with ", str(num_pts), " points.")
    for i, pd in enumerate(all_configs[num_pts]):
        print("\nOn {}/{}".format(i+1, len(all_configs[num_pts])))
        # for l in pd.lines:
        #     print(l)
        is_possible = possibly_embeddable(pd)
        if is_possible:
            print("ADDED POSSIBLY EMBEDDABLE")
            possibly_embeddable_pds.append(pd)

# for num_pts in range(7, 17):
#     print("On designs with ", str(num_pts), " points.")
#     for i, pd in enumerate(all_configs[num_pts]):
#         print("\n\nOn {}/{}".format(i+1, len(all_configs[num_pts])))
#         init, fo = find_forcing_fixture(pd)
#         if len(init) > 5:
#             print("COULD NOT FIND FIXTURE OF SIZE 5")
#             for l in pd.lines:
#                 print(l)
#             continue

#         pt_coords = make_pt_coords(pd, fo)
#         zero_eqs = make_eqs(pd, pt_coords)
#         if zero_eqs == False:
#             continue
#         print("Equations:")
#         for e in zero_eqs:
#             print(e)