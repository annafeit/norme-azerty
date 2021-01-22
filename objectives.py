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

import read_input
import numpy as np
import os
import codecs
from gurobipy import *

PYTHONIOENCODING = "utf-8"

capitals = {u'à': u'À', u'â': u'Â', u'ç': u'Ç', u'é': u'É', u'è': u'È', u'ê': u'Ê',
            u'ë': u'Ë', u'î': u'Î', u'ï': u'Ï', u'ô': u'Ô',
            u'ù': u'Ù', u'û': u'Û', u'ü': u'Ü', u'ÿ': u'Ÿ', u'æ': u'Æ', u'œ': u'Œ',
            u'ß': u'ẞ', u'þ': u'Þ', u'ð': u'Ð', u'ŋ': u'Ŋ', u'ĳ': u'Ĳ',
            u'ə': u'Ə', u'ʒ': u'Ʒ', u'ı': u'İ'}


def get_objectives(mapping, w_p, w_a, w_f, w_e, corpus_weights, quadratic=0):
    """
        Computes and returns the objectives of the given mapping with the given weights
    """
    azerty, \
    characters, \
    keyslots, \
    letters, \
    p_single, p_bigram, \
    performance, \
    similarity_c_c, similarity_c_l, \
    distance_level_0, distance_level_1, \
    ergonomics \
        = read_input.get_all_input_values(corpus_weights)

    return accu_get_objectives(mapping, w_p, w_a, w_f, w_e,
                               azerty,
                               characters,
                               keyslots,
                               letters,
                               p_single, p_bigram,
                               performance,
                               similarity_c_c, similarity_c_l,
                               distance_level_0, distance_level_1,
                               ergonomics, quadratic=quadratic)


def accu_get_objectives(mapping, w_p, w_a, w_f, w_e,
                        azerty,
                        characters,
                        keyslots,
                        letters,
                        p_single, p_bigram,
                        performance,
                        similarity_c_c, similarity_c_l,
                        distance_level_0, distance_level_1,
                        ergonomics, quadratic=1):
    """
        For a given mapping, returns the objective value for the given weights, and the individual objectives values for P,A,F,E
    """
    # Compute linear cost matrices
    linear_cost, x_P, x_A, x_F, x_E = get_linear_costs(w_p, w_a, w_f, w_e,
                                                       azerty,
                                                       characters,
                                                       keyslots,
                                                       letters,
                                                       p_single, p_bigram,
                                                       performance,
                                                       similarity_c_c, similarity_c_l,
                                                       distance_level_0, distance_level_1,
                                                       ergonomics)

    # remove letters from mapping that are not in characters list

    fixed = read_input.get_fixed_mapping()
    remove_keys = []
    # remove invalid characters and keyslots from the mapping
    for c, s in mapping.items():
        if not c in characters:
            remove_keys.append(c)
            print("%s not in the to-be-mapped character set" % c)
        elif not s in keyslots:
            remove_keys.append(c)
            print("%s not mapped to a keyslot for which we have values" % s)
    mapping = {c: s for c, s in mapping.items() if c not in remove_keys}

    P = 0
    A = 0
    F = 0
    E = 0
    for c, s in mapping.items():
        P += x_P[c, s]
        A += x_A[c, s]
        F += x_F[c, s]
        E += x_E[c, s]

    lin_A = A

    if quadratic:
        prob_sim, distance_level_0_norm = get_quadratic_costs(characters, \
                                                              keyslots, \
                                                              p_single,
                                                              similarity_c_c)
        for (c1, c2) in similarity_c_c:
            if c1 in mapping and c2 in mapping:
                s1 = mapping[c1]
                s2 = mapping[c2]
                v = prob_sim[c1, c2] * distance_level_0_norm[s1, s2]
                A += v

    if P < 0:
        print("Performance negative, rounded to 0: %f" % P)
        P = np.max([0, P])
    if A < 0:
        print("Association negative, rounded to 0: %f" % A)
        A = np.max([0, A])
    if F < 0:
        print("Familiarity negative, rounded to 0: %f" % F)
        F = np.max([0, F])
    if E < 0:
        print("Ergonomics negative, rounded to 0: %f" % E)
        E = np.max([0, E])
    objective = w_p * P + w_a * A + w_f * F + w_e * E
    print("objective: ", objective)
    return objective, P, A, F, E


def get_linear_costs(w_p, w_a, w_f, w_e,
                     azerty,
                     characters,
                     keyslots,
                     letters,
                     p_single, p_bigram,
                     performance,
                     similarity_c_c, similarity_c_l,
                     distance_level_0, distance_level_1,
                     ergonomics):
    """ computes the linear cost: for each linear variable x[c,s] compute the P, A, F and E term (as if it is chosen)
        Returns the linear cost for each objective and the weighted sum.   
    """
    scenario, char_set = read_input.get_scenario_and_char_set()

    if os.path.isfile("input/normalized/" + "x_P_" + scenario + "_" + char_set + ".txt"):
        x_P = _read_tuple_list_to_dict("input/normalized/" + "x_P_" + scenario + "_" + char_set + ".txt")
        x_A = _read_tuple_list_to_dict("input/normalized/" + "x_A_" + scenario + "_" + char_set + ".txt")
        x_F = _read_tuple_list_to_dict("input/normalized/" + "x_F_" + scenario + "_" + char_set + ".txt")
        x_E = _read_tuple_list_to_dict("input/normalized/" + "x_E_" + scenario + "_" + char_set + ".txt")
    else:

        print("Getting linear cost")
        x_P = {}
        x_A = {}
        x_F = {}
        x_E = {}

        for c in characters:
            for s in keyslots:
                P = 0
                A = 0
                # if that character was previously not on azerty, distance is 0.
                F = p_single[c] * distance_level_1.get((s, azerty.get(c, "NaN")), 0)
                E = 0
                for l in letters:
                    # update performance
                    if (c, l) in p_bigram:
                        P += (p_bigram[(c, l)] * performance[(s, azerty[l])])
                    if (l, c) in p_bigram:
                        P += (p_bigram[(l, c)] * performance[(azerty[l], s)])
                        # update association. This is symmetric, so we add it twice to make it comparable with the other scores
                    if (c, l) in similarity_c_l or (l, c) in similarity_c_l:
                        try:
                            A += 2 * (similarity_c_l[(c, l)] * distance_level_0[s, azerty[l]])
                        except KeyError:
                            A += 2 * (similarity_c_l[(l, c)] * distance_level_0[s, azerty[l]])
                            # update ergonomics
                    if (c, l) in p_bigram:
                        E += (p_bigram[(c, l)] * ergonomics[(s, azerty[l])])
                    if (l, c) in p_bigram:
                        E += (p_bigram[(l, c)] * ergonomics[(azerty[l], s)])

                x_P[c, s] = P
                x_A[c, s] = A

                x_F[c, s] = F
                x_E[c, s] = E

        # now normalize these terms by minimizing/maximizing each individually such that they are all between 0 and 1
        print("========= Normalize Performance =========")
        x_P = normalize_empirically(x_P, characters, keyslots, capitalization_constraints=1)
        print("========= Normalize Association =========")
        x_A = normalize_empirically(x_A, characters, keyslots, capitalization_constraints=1)
        print("========= Normalize Familiarity =========")
        x_F = normalize_empirically(x_F, characters, keyslots, capitalization_constraints=1)
        print("========= Normalize Ergonomics =========")
        x_E = normalize_empirically(x_E, characters, keyslots, capitalization_constraints=1)

        # write into file for later use
        write_tuplelist(x_P, "input/normalized/" + "x_P_" + scenario + "_" + char_set + ".txt")
        write_tuplelist(x_A, "input/normalized/" + "x_A_" + scenario + "_" + char_set + ".txt")
        write_tuplelist(x_F, "input/normalized/" + "x_F_" + scenario + "_" + char_set + ".txt")
        write_tuplelist(x_E, "input/normalized/" + "x_E_" + scenario + "_" + char_set + ".txt")

    # weighted sum of linear terms
    linear_cost = {}
    for c in characters:
        for s in keyslots:
            linear_cost[c, s] = (w_p * x_P[c, s]) + (w_a * x_A[c, s]) + (w_f * x_F[c, s]) + (w_e * x_E[c, s])

    return linear_cost, x_P, x_A, x_F, x_E


def get_quadratic_costs(characters, \
                        keyslots, \
                        p_single, \
                        similarity_c_c):
    scenario, char_set = read_input.get_scenario_and_char_set()
    if os.path.isfile("input/normalized/" + "prob_sim_" + scenario + "_" + char_set + ".txt"):
        prob_sim = _read_tuple_list_to_dict("input/normalized/" + "prob_sim_" + scenario + "_" + char_set + ".txt")
        distance_level_0_norm = _read_tuple_list_to_dict(
            "input/normalized/" + "distance_" + scenario + "_" + char_set + ".txt")
    else:
        distance_level_0 = read_input.get_distance_consistency()
        print("Getting quadratic cost")
        prob_sim = {}
        for c1 in characters:
            for c2 in characters:
                if (c1, c2) in similarity_c_c.keys():
                    # do not add association cost if both lowercase and capital letter are
                    # in character set, will be accounted for by cap.constr.
                    if not (c1 in capitals and c2 == capitals[c1]):
                        p = similarity_c_c[c1, c2]
                        prob_sim[(c1, c2)] = p
                    else:
                        prob_sim[(c1, c2)] = 0
                else:
                    prob_sim[(c1, c2)] = 0

        # normalize with normalization factor of full objective (later multiplied with distance)
        max_sum = 0
        min_sum = 0
        for c1 in characters:
            # for each character determine the maximum association cost for assigning that character to a slot and sum up
            costs_per_slot_min = []
            costs_per_slot_max = []
            for s1 in keyslots:
                tmp_sum_min = 0  # sum up the association cost for all other characters
                tmp_sum_max = 0
                for c2 in characters:
                    if c1 != c2:
                        # add maximum association cost if that character was assigned to a key
                        tmp_sum_max += np.max(
                            [prob_sim[c1, c2] * distance_level_0[s1, s2] for s2 in keyslots if s2 != s1])
                        tmp_sum_min += np.min(
                            [prob_sim[c1, c2] * distance_level_0[s1, s2] for s2 in keyslots if s2 != s1])
                costs_per_slot_min.append(tmp_sum_min)
                costs_per_slot_max.append(tmp_sum_max)
            max_sum += np.max(costs_per_slot_max)  #
            min_sum += np.min(costs_per_slot_min)  #

        # normalization factor is included in the distance because there all values are > 0. Otherwise there are some problems
        distance_level_0_norm = distance_level_0.copy()
        n = len(characters)
        for (s1, s2), v in distance_level_0.items():
            if v > 0:
                distance_level_0_norm[(s1, s2)] = ((v - (min_sum / float(n))) / (float(max_sum) - float(min_sum)))

        # write into file for later use
        write_tuplelist(prob_sim, "input/normalized/" + "prob_sim_" + scenario + "_" + char_set + ".txt")
        write_tuplelist(distance_level_0_norm, "input/normalized/" + "distance_" + scenario + "_" + char_set + ".txt")

    return prob_sim, distance_level_0_norm


def normalize_dict_values(d):
    """
    Normalizes all values to be between 0 and 1 such that they maximally sum up to 1
    """
    # normalize single values to be between 0 and 1
    maximum = np.max(list(d.values()))
    minimum = np.min(list(d.values()))

    for k, v in d.items():
        d[k] = (v - minimum) / float(maximum - minimum)
    return d


def normalize_empirically(X_O, characters, keyslots, capitalization_constraints=1):
    """
     Normalize empirically for the result to be between 0 and 1. 
     First minimizes and maximizes the keyboard problem for the given cost (X_O) and then normalizes all values in X_O
     to minimally/maximally sum up to 0/1. 
     Can only be used for the linear terms (Performance, Association, Ergonomics, Familiarity)
    """

    if len(characters) > len(keyslots):
        print("Error: more characters sthan keyslots")
        return

    m = Model("keyboard_layout")
    # add decision variables
    x = {}
    for c in characters:
        for s in keyslots:
            n = u"" + c + u"_to_" + s
            n = n.encode("utf-8")
            x[c, s] = m.addVar(vtype=GRB.BINARY, name=n)

    m.update()
    m._vars = m.getVars()

    # Define the objective terms
    O = quicksum(
        X_O[c, s] * x[c, s] \
        for c in characters for s in keyslots
    )

    m._O = O

    # add the constraints. One for each character, one for each keyslot
    for c in characters:
        m.addConstr(quicksum(x[c, s] for s in keyslots) == 1, c + "_mapped_once")

    for s in keyslots:
        m.addConstr(quicksum(x[c, s] for c in characters) <= 1, s + "_assigned_at_most_once")

    if capitalization_constraints:
        print("Adding capitalization constraints")
        for c, s_c in capitals.items():
            if c in characters and s_c in characters:
                for k in keyslots:
                    if "Shift" in k:
                        m.addConstr(x[c, k] == 0, c + "_not_mapped_to_shifted_key_" + k)
                    else:
                        if k + "_Shift" in keyslots:
                            # if character is assigned to this key, its capital version must be assigned to shifted version of the key
                            m.addConstr(x[c, k] - x[s_c, k + "_Shift"] == 0,
                                        c + "_and_" + s_c + "mapped_to_shifted_" + k)
                        else:
                            # unshifted version should not be assigned to key where shifted version is not available
                            m.addConstr(x[c, k] == 0, c + "_no_shift_available_" + k)

    m.setParam("OutputFlag", 0)
    m.update()

    # Set objective    
    m.setObjective(O, GRB.MINIMIZE)
    m.update()
    # Optimize
    m.optimize()
    minimum = m._O.getValue()
    print("===> Minimum: %.5f" % minimum)

    # Set objective
    m.setObjective(O, GRB.MAXIMIZE)
    m.update()
    # Optimize
    m.optimize()
    maximum = m._O.getValue()
    print("===> Maximum: %.5f" % maximum)

    n = len(characters)
    for k, v in X_O.items():
        X_O[k] = (v - (minimum / float(n))) / (float(maximum) - float(minimum))

    return X_O


def _read_tuple_list_to_dict(path):
    """
        Reads a file into a dictionary. 
        The file must have the following format:
        key1 key2 value
        Then the dictionary is of the form:
        {(key1,key2):value}
    """
    with codecs.open(path, 'r', encoding="utf-8") as bigram_file:
        all_lines = bigram_file.readlines()
        lines = [l.rstrip() for l in all_lines]
        # create dict
        p_bigrams = {}
        for l in lines:
            parts = l.split(" ")
            if len(parts) == 3:
                if parts[0] != "" and parts[1] != "":
                    p_bigrams[(parts[0], parts[1])] = float(parts[2])
    return p_bigrams


def write_tuplelist(data, filename):
    data_strings = ["%s %s %f\n" % (s, l, n) for (s, l), n in data.items()]
    data_strings = [s.encode("utf-8") for s in data_strings]
    with open(filename, 'w') as data_file:
        data_file.writelines(data_strings)
