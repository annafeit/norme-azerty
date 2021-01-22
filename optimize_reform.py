# -*- coding: utf-8 -*-
#!/usr/bin/env python3.8

# Copyright (c) 2019 Anna Maria Feit
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from gurobipy import *
import numpy as np
from objectives import *
from read_input import *
from kaufmanbroeckx import *
from mapping import *
from plotting import *

PYTHONIOENCODING = "utf-8"

directory = "mappings/"
firstline = "#"
filename = "solution"
round = 1


def simple_mst_writer(model, mstfilename, nodecnt, obj, bound):
    mstfile = open(mstfilename, 'w')
    soln = model.cbGetSolution(model._vars)

    mstfile.write(firstline)  # add first line with weights and scenario
    gap = np.abs((100 * (bound - obj)) / obj)
    mstfile.write(
        '# MIP start from soln at node %d, Objective %.8f, bound %.8f, gap %.3f\n' % (nodecnt, obj, bound, gap))
    for n, soln in zip(model._varNames, soln):
        if "x" in n:  # only save x variables
            mstfile.write('%s %i\n' % (n, soln))
    mstfile.close()


def opti_callback(model, where):
    """
    Writes intermediate solutions
    """
    try:
        if where == GRB.callback.MIPSOL:
            obj = model.cbGet(GRB.callback.MIPSOL_OBJ)
            bound = model.cbGet(GRB.callback.MIPSOL_OBJBND)
            nodecnt = int(model.cbGet(GRB.callback.MIPSOL_NODCNT))
            print("Found incumbent soln at node %i objective %f" % (nodecnt, obj))
            simple_mst_writer(model, directory + filename + "_R" + str(round) + '_%f.mst' % obj, nodecnt, obj, bound)
    except GurobiError as e:
        print("Gurobi Error:")
        print(e.errno)
        print(e.message)


def optimize(w_p, w_a, w_f, w_e, corpus_weights, scenario, char_set, startsolution="", fixation_constraints=1,
             coherence_constraints=1, capitalization_constraints=1):
    """
    Optimizes the pre-defined scenario with the given weights.

    Parameters:
        w_p, w_a, w_f, w_e: weights for the individual objectives. Must sum up to one
        corpus_weights: a dict of corpus identifiers to weights, weights should sum up to one
        scenario, char_set: the scenario that defines the fixed characters and the character set of to-be-assigned chars
        startsolution: optional, give a solution to start from to speed up search or continue from a previous search.
        fixation_constraints, coherence_constraints, capitalization_constraints: optional, turning additional constraints
            on or off.
    """
    set_scenario_files(scenario, char_set)

    print("Create optimization lp file")
    # create the the input file for the kaufmannbroeckx reformulation
    reform_filename = scenario + char_set + "_reform"
    create_reformulation_input(w_p, w_a, w_f, w_e, corpus_weights, reform_filename)

    # reformulate and create the lp file for Gurobi
    kaufmannbroeckx_reformulation("mappings/reformulations/" + reform_filename + ".txt",
                                  "mappings/reformulations/" + reform_filename + ".lp")

    # add additional constraints
    new_lp = add_constraints("mappings/reformulations/" + reform_filename + ".lp",
                             fixation_constraints, coherence_constraints, capitalization_constraints)

    # optimize
    return optimize_reformulation(new_lp, scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights,
                                  startsolution=startsolution, )


def optimize_reformulation(lp_path, scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights, startsolution="",
                           mipfocus=0):
    """
    Optimizes the given lp file
    Logs intermediate solutions
    Optional: give a solution to start from to speed up search or continue from a previous search.
        Set optional Gurobi parameters
    """
    new_path = lp_path
    global round
    global filename
    filename = lp_path.split("/")[-1]
    filename = filename[:-3]  # remove .lp
    print("filename: " + filename)

    # get the parameters of the first line
    # global firstline
    # f = lp_path[:-3] #remove .lp
    # if f.split("_")[-1] == "constrained" or f.split("_")[-1] == "capitalrelaxed":
    #    f = "_".join(f.split("_")[:-1])
    # reformulationFile = open(f+".txt",'r')
    # line = reformulationFile.readline().strip()
    # capitalization = f.split("_")[-1]
    # firstline= line + ",capitalization=%s\n"%capitalization #save first line with weights and scenario

    # add folder for logging intermediate solutions
    global directory
    directory = "mappings/" + filename.split("_")[0] + "/"
    print("directory: " + directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model = read(new_path)
    model.update()

    model.setParam("NodefileStart", 0.5)
    model.setParam("MIPFocus", mipfocus)
    print("optimizing...")

    if startsolution != "":
        model.read(startsolution)
    model._vars = model.getVars()
    model._varNames = [v.varName for v in model.getVars()]

    try:
        model.optimize(opti_callback)
    except GurobiError as e:
        print("Gurobi Error:")
        print(e.errno)
        print(e.message)

    # Output objective values:
    if model.status == GRB.Status.OPTIMAL:
        print('Optimal objective: %g' % model.objVal)
        writeSolution(model)
        mapping = evaluate_optimized_reformulation(scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights)

    elif model.status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        print('Saved suboptimal solution with objective: %g' % model.objVal)
        writeSolution(model)
        mapping = None

    else:
        # there is some solution though not the optimal one, return that one
        mapping = evaluate_optimized_reformulation(scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights)

    return model, mapping


def add_constraints(lp_path, fixation_constraints=1, coherence_constraints=1, capitalization_constraints=1):
    """'
    1. add fixation constraints:
        characters that are defined in the fixed mapping but also part of the characterset must be enforced on the defined key
    2. add coherence constraint:
        enclosing characters placed horizontally next to each other with same modifier.
        Note, the corresponding character pairs are defined in read_input.py via the function get_coherence_pairs()
    3. add capitalization constraints: 
        for each key, if the shifted version of that key is also available, 
        enforce that for all characters mapped to that key, the capital version of the character is mapped to the shifted key
        if the shifted version is not available as a free slot, or this is a shifted version, then enforce that the letter 
        is not mapped to that key.
    """
    lp_original = open(lp_path, 'r')

    new_path = ".".join(lp_path.split(".")[:-1]) + "_constrained." + lp_path.split(".")[-1]
    new_lp = open(new_path, 'w')

    all_lines = lp_original.readlines()
    for i in range(0, len(all_lines)):
        new_lp.write(all_lines[i])
        if i < len(all_lines) - 1:
            # if next line is the "binaries" line, add capitalization constraints here
            if "binaries" in all_lines[i + 1]:
                keyslots = get_keyslots()
                characters = get_characters()
                fixed = get_fixed_mapping()
                coherence_pairs = get_coherence_pairs()
                # 1. add constraints for fixed characters:
                if fixation_constraints:
                    print("Adding fixation constraints")
                    for c, k in fixed.items():
                        if c in characters:
                            c_index = characters.index(c)
                            k_index = keyslots.index(k)
                            s = "x(%i,%i) = 1\n" % (c_index, k_index)
                            new_lp.write(s)

                # 2. Add constraints for consistency
                if coherence_constraints:
                    print("Adding coherence constraints")
                    for k_index in range(0, len(keyslots)):
                        k = keyslots[k_index]
                        for (i, j) in coherence_pairs:
                            if i in characters and j in characters:
                                i_index = characters.index(i)
                                j_index = characters.index(j)
                                # they should be placed directly next to each other
                                # determine key next to k in same row with same modifier
                                l_num = int(k[1:3]) + 1
                                if l_num < 10:
                                    l_num = "0%i" % l_num
                                else:
                                    l_num = "%i" % l_num
                                l = k[0] + l_num + k[3:]
                                if l in keyslots:
                                    l_index = keyslots.index(l)
                                    s = "x(%i,%i) - x(%i,%i) = 0\n" % (i_index, k_index, j_index, l_index)
                                    new_lp.write(s)
                                else:
                                    # no key next to it available, do not assign here
                                    s = "x(%i,%i) = 0\n" % (i_index, k_index)
                                    new_lp.write(s)
                # 3. Add capitalization constraints normal
                if capitalization_constraints:
                    print("Adding capitalization constraints")
                    for k_index in range(0, len(keyslots)):
                        k = keyslots[k_index]

                        if "Shift" in k:
                            # shifted key do not assign lowercase here:
                            for c, s_c in capitals.items():
                                if c in characters and s_c in characters:  # only if the capital version is in the to-be-mapped characterset
                                    s = "x(%i,%i) = 0\n" % (c_index, k_index)
                                    new_lp.write(s)

                                    unshifted = ("_").join(k.split("_")[:-1])
                                    if unshifted not in keyslots:
                                        # do not assign uppercase letter here neither, because unshifted is not available
                                        s_index = characters.index(s_c)
                                        s = "x(%i,%i) = 0\n" % (c_index, k_index)
                                        new_lp.write(s)




                        else:
                            # non-shifted key, do not assign uppercase here:
                            for c, s_c in capitals.items():
                                if c in characters and s_c in characters:  # only if the capital version is in the to-be-mapped characterset
                                    s_index = characters.index(s_c)
                                    s = "x(%i,%i) = 0\n" % (s_index, k_index)
                                    new_lp.write(s)

                            if k + "_Shift" in keyslots:
                                # if character is assigned to this key, its capital version must be assigned to shifted version of the key
                                for c, s_c in capitals.items():
                                    if c in characters and s_c in characters:  # only if the capital version is in the to-be-mapped characters
                                        k_shifted_index = keyslots.index(k + "_Shift")
                                        c_index = characters.index(c)
                                        s_index = characters.index(s_c)
                                        # x(i,k) - x(j,l) = 0
                                        s = "x(%i,%i) - x(%i,%i) = 0\n" % (c_index, k_index, s_index, k_shifted_index)
                                        new_lp.write(s)
                            else:
                                # no shfited version of k available, do not assign lowercase letter here neither
                                for c, s_c in capitals.items():
                                    if c in characters and s_c in characters:  # only if the capital version is in the to-be-mapped characters
                                        s = "x(%i,%i) = 0\n" % (c_index, k_index)
                                        new_lp.write(s)

    new_lp.close()
    lp_original.close()
    return new_path


def writeSolution(model):
    """
    Writes a solution after stopping an optimization
    """
    path = directory + "sub_" + filename + "_%f.mst" % model.objVal
    model.write(path)
    f = open(path, 'r')
    alllines = f.readlines()
    f.close()

    newf = open(path, 'w')
    newf.write(firstline)
    gap = np.abs((100 * (model.objBound - model.objVal)) / model.objVal)
    newf.write('# Objective %.8f, bound %.8f, gap %.3f\n' % (model.objVal, model.objBound, gap))
    newf.writelines(alllines)
    newf.close()


def create_reformulation_input(w_P, w_A, w_F, w_E, corpus_weights, filename, quadratic=1):
    """
        creates the file reformulation_input.txt which is used as input for the kaufmann-broeckx reformulation        
    """

    # Read in input values

    azerty, \
    characters, \
    keyslots, \
    letters, \
    p_single, p_bigram, \
    performance, \
    similarity_c_c, similarity_c_l, \
    distance_level_0, distance_level_1, \
    ergonomics \
        = get_all_input_values(corpus_weights)

    # linear_costs is already weighted, the x_ are not
    linear_costs, x_p, x_a, x_f, x_e = get_linear_costs(w_P, w_A, w_F, w_E,
                                                        azerty,
                                                        characters,
                                                        keyslots,
                                                        letters,
                                                        p_single, p_bigram,
                                                        performance,
                                                        similarity_c_c, similarity_c_l,
                                                        distance_level_0, distance_level_1,
                                                        ergonomics)

    # modify linear cost such that they are non-negative. Add to each term the absolute minimum number (if negative).
    # So this will be added to the overall cost (on the diagional of the quadratic matrix) and thus the optimum stays
    # the same and the individual costs won't change. The overall cost is n*minimum larger
    minimum = min(0, np.min(list(linear_costs.values())))

    linear_costs = {(c, s): (v + np.abs(minimum)) for (c, s), v in linear_costs.items()}

    scenario, char_set = get_scenario_and_char_set()
    # Writes an input file for the reformualtion
    f = codecs.open("mappings/reformulations/" + filename + ".txt", 'w', encoding="utf-8")
    f.write(
        "#scenario=%s,set=%s,w_P=%f,w_A=%f,w_F=%f,w_E=%f,w_formal=%f,w_twitter=%f,w_code=%f\n" % (scenario, char_set,
                                                                                                  w_P, w_A, w_F, w_E,
                                                                                                  corpus_weights[
                                                                                                      "formal"],
                                                                                                  corpus_weights[
                                                                                                      "twitter"],
                                                                                                  corpus_weights[
                                                                                                      "code"]))
    f.write("# number of letters and keys\n")
    f.write(str(len(keyslots)) + "\n")
    f.write("# w_A*probabilities*similarities\n")

    ## QUADRATIC PART
    # this is unweighted
    prob_sim, distance_level_0_norm = get_quadratic_costs(
        characters,
        keyslots,
        p_single,
        similarity_c_c)

    for c1 in characters:
        prob_strings = []
        for c2 in characters:
            prob_strings.append("%.12f" % (quadratic * w_A * prob_sim[(c1, c2)]))  # remember to weight
        # add dummy values to fill it up to number of keyslots
        for i in range(len(keyslots) - len(characters)):
            prob_strings.append("0")
        f.write(" ".join(prob_strings) + "\n")
    # add dummy values to fill it up to number of keyslots
    for i in range(len(keyslots) - len(characters)):
        prob_strings = []
        for c2 in characters:
            prob_strings.append("0")
        # add dummy values to fill it up tp number of keyslots
        for i in range(len(keyslots) - len(characters)):
            prob_strings.append("0")
        f.write(" ".join(prob_strings) + "\n")

    # write the distances for quadratic part only!
    f.write("# distances\n")
    distances = distance_level_0_norm

    for s1 in keyslots:
        dist_strings = []
        for s2 in keyslots:
            d = distances[(s1, s2)]
            dist_strings.append("%.12f" % d)

        f.write(" ".join(dist_strings) + "\n")

    f.write("# fixation of the spacebar to the bottom\n")
    f.write("0\n")
    f.write("# scale for rounding down the probabilities\n")
    f.write("1e14\n")

    ## LINEAR PART
    f.write("# linear cost\n")

    for c in characters:
        lin_strings = []
        for s in keyslots:
            l = linear_costs[(c, s)]
            lin_strings.append("%.12f" % l)

        f.write(" ".join(lin_strings) + "\n")
    # add dummy values to fill it up to number of keyslots
    for i in range(len(keyslots) - len(characters)):
        lin_strings = []
        for s in keyslots:
            lin_strings.append("0")
        f.write(" ".join(lin_strings) + "\n")

    f.close()


def evaluate_optimized_reformulation(scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights, quadratic=1):
    """
        Searches for the last (=best) mapping produced by the solver, stores a human-readable format,
        computes its objective values and plots it.
    """

    # find the newest mapping
    newest = max(glob.iglob("mappings/" + scenario + char_set + "/*.mst"), key=os.path.getctime)
    # save in human-readable format
    mapping = create_map_from_reformulation(newest)
    log_mapping(mapping, ".".join(newest.split(".")[0:-1]) + ".txt")

    # create plot
    plot_mapping(mapping, plotname=newest + ".png", w_p=w_p, w_a=w_a, w_f=w_f, w_e=w_e,
                 corpus_weights=corpus_weights, quadratic=1)

    return mapping
