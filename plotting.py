# -*- coding: utf-8 -*-

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

from __future__ import unicode_literals

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import cm
import codecs
from objectives import *
from read_input import *
from mapping import *
import re

PYTHONIOENCODING = "utf-8"

# box dimensions
key_height = 4
key_width = 4

# keyboard specifics
row_distance = 0.5
column_distance = 0.5
row_shift = {'A': 0, 'B': 0, 'C': key_width / 2, 'D': key_width, 'E': 3 * key_width / 2}
row_numbers = {u"A": 0, u"B": 1, u"C": 2, u"D": 3, u"E": 4}

fixed_color = 'gray'
special_color = "#4C72B0"
dead_color = "#C44E52"
fixed_font = 'normal'
special_font = 'bold'

# text positions
pos_normal_x = 0.5
pos_normal_y = 0.5
pos_shift_x = 0.5
pos_shift_y = key_height - 0.5
pos_alt_x = key_width - 0.5
pos_alt_y = 0.5
pos_alt_shift_x = key_width - 0.5
pos_alt_shift_y = key_height - 0.5


def swap_and_plot(mapping, char1, char2, corpus_weights, w_p, w_a, w_f, w_e, plot=True):
    # read in mapping
    if type(mapping) == str:
        mapping = get_mapping(mapping)

    if char1 not in mapping:
        print("%s not in mapping" % char1)
        return
    if char2 not in mapping:
        print("%s not in mapping" % char2)
        return

    # create swapped mapping
    new_mapping = mapping.copy()
    new_mapping[char1] = mapping[char2]
    new_mapping[char2] = mapping[char1]

    plot_mapping_comparison(mapping, new_mapping, corpus_weights, w_p, w_a, w_f, w_e, plot=True)
    return new_mapping


def plot_mapping_comparison(mapping, new_mapping, corpus_weights, w_p, w_a, w_f, w_e, plot=True, _axes=[],
                            subplot=[-1, -1, -1]):
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

    prob_sim_matrix, distance_level_0_norm = get_quadratic_costs(characters,
                                                                 keyslots,
                                                                 p_single,
                                                                 similarity_c_c)

    # Function to evaluate a mapping
    def evaluate_mapping(new_map):
        P = 0
        A = 0
        F = 0
        E = 0
        for c, s in new_map.items():
            P += x_P[c, s]
            A += x_A[c, s]
            F += x_F[c, s]
            E += x_E[c, s]
        for (c1, c2) in similarity_c_c:
            if c1 in new_map and c2 in new_map:
                s1 = new_map[c1]
                s2 = new_map[c2]
                v = prob_sim_matrix[c1, c2] * distance_level_0_norm[s1, s2]
                A += v
        if P < 0:
            print("Performance negative, rounded to 0: %f" % P)
            P = np.maximum(0, P)
        if A < 0:
            print("Association negative, rounded to 0: %f" % A)
            A = np.maximum(0, A)
        if F < 0:
            print("Familiarity negative, rounded to 0: %f" % F)
            F = np.maximum(0, F)
        if E < 0:
            print("Ergonomics negative, rounded to 0: %f" % E)
            E = np.maximum(0, E)
        objective = w_p * P + w_a * A + w_f * F + w_e * E
        return objective, P, A, F, E

    objective, P, A, F, E = evaluate_mapping(mapping)
    new_objective, new_P, new_A, new_F, new_E = evaluate_mapping(new_mapping)

    plot_mapping(new_mapping, corpus_weights=corpus_weights,
                 w_p=w_p, w_a=w_a, w_f=w_f, w_e=w_e,
                 p=new_P, a=new_A, f=new_F, e=new_E, objective=new_objective, axes=_axes, subplot=subplot[0])

    if plot:
        if not len(_axes) > 1:
            fig, axes = plt.subplots(1, 2)
            fig.set_size_inches(10, 5)
            fig.tight_layout()
        else:
            axes = [_axes[subplot[1]], _axes[subplot[2]]]
        w = 0.8

        axes[1].bar([0, 1, 2, 3, 4],
                    [objective - new_objective, P - new_P, A - new_A, F - new_F, E - new_E], width=w)

        axes[1].set_xticks([0 + w / 2, 1 + w / 2, 2 + w / 2, 3 + w / 2, 4 + w / 2])
        axes[1].set_xticklabels(["obj", "P", "A", "F", "E"])
        axes[1].set_ylabel("Absolute difference")

        axes[0].bar([0, 1, 2, 3, 4],
                    [100 * (objective - new_objective) / objective, 100 * (P - new_P) / P, 100 * (A - new_A) / A,
                     100 * (F - new_F) / F, 100 * (E - new_E) / E], width=w)

        axes[0].set_xticks([0 + w / 2, 1 + w / 2, 2 + w / 2, 3 + w / 2, 4 + w / 2])
        axes[0].set_xticklabels(["obj", "P", "A", "F", "E"])
        axes[0].set_ylabel("Relative difference (%)")

        print("Comparison - overall: %f, performance: %f, association: %f, familiarity: %f, ergonomics %f" % (
            100 * (objective - new_objective) / objective, 100 * (P - new_P) / P, 100 * (A - new_A) / A,
            100 * (F - new_F) / F, 100 * (E - new_E) / E))


def plot_mapping(mapping, plotname="", azerty=-1, numbers=-1, letters=-1,
                 corpus_weights=-1, quadratic=1,
                 objective=-1,
                 p=-1, a=-1, f=-1, e=-1, w_p=-1, w_a=-1, w_f=-1, w_e=-1, print_keycode=0,
                 axes=[], subplot=-1):
    """
    Plots the given mapping.
    Mapping can be a path to the mapping file, created by the reformulation, or an actual mapping. 
    If no objective is given, it computes the objective values.
    """
    if type(mapping) == str:
        mapping = get_mapping(mapping)

    if objective == -1:
        objective, p, a, f, e = get_objectives(mapping, w_p, w_a, w_f, w_e, corpus_weights, quadratic=quadratic)

    if azerty == -1:
        azerty = get_azerty()
    if numbers == -1:
        numbers = get_fixed_mapping()
    if letters == -1:
        letters = get_letters()

    with open('input/keyslots/all_slots.txt') as file:
        all_slots = file.read().splitlines()

    if len(axes) > 1:
        if subplot != -1:
            ax = axes[subplot]
        else:
            ax = axes
    else:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 4)

    for slot in all_slots:
        row = row_numbers[slot[0]]
        column = int(slot[1:3])
        level = slot[4:]

        if level == "":
            height = key_height
            width = key_width
            # Space
            if row == 0 and column == 3:
                width = key_width * 5 + 4 * row_distance

            x = (column * key_width) + column * column_distance - row_shift[slot[0]]
            y = (row * key_height) + row * row_distance

            ax.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    width,  # width
                    height,  # height
                    fill=False
                )
            )
            if print_keycode:
                ax.text(x + (key_width * 0.5), y + (key_height * 0.35), slot,
                        horizontalalignment='center',
                        fontsize=8,
                        color=(0.8, 0.8, 0.8)
                        )

    # Add fixed character annotation
    for l in letters:
        if not l == "space":
            slot = azerty[l]
            row = row_numbers[slot[0]]
            column = int(slot[1:3])

            l = l.capitalize()

            # capital letters on Shifted level
            pos_x = pos_shift_x
            pos_y = pos_shift_y
            ha = 'left'
            va = 'top'
            x = (column * key_width) + column * column_distance + pos_x - row_shift[slot[0]]
            y = (row * key_height) + row * row_distance + pos_y

            ax.text(x, y, l,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    fontsize=12,
                    fontweight=fixed_font,
                    color=fixed_color  # (0.4,0.4,0.4)
                    )

    for l in numbers:
        r = re.compile("[A-Z]")
        if not r.findall(l):
            slot = numbers[l]
            row = row_numbers[slot[0]]
            column = int(slot[1:3])
            level = slot[4:]

            if level == "":
                pos_x = pos_normal_x
                pos_y = pos_normal_y
                ha = 'left'
                va = 'bottom'
            if level == "Shift":
                pos_x = pos_shift_x
                pos_y = pos_shift_y
                ha = 'left'

                va = 'top'
            if level == "Alt":
                pos_x = pos_alt_x
                pos_y = pos_alt_y
                ha = 'right'
                va = 'bottom'
            if level == "Alt_Shift":
                pos_x = pos_alt_shift_x
                pos_y = pos_alt_shift_y
                ha = 'right'
                va = 'top'

            if u"d" in l and len(l) > 1:  # dead key
                l = re.sub('d', '', l)
                c = dead_color
            else:
                c = fixed_color

            x = (column * key_width) + column * column_distance + pos_x - row_shift[slot[0]]
            y = (row * key_height) + row * row_distance + pos_y

            ax.text(x, y, l,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    fontsize=10,
                    fontweight=fixed_font,
                    color=c  # (0.4,0.4,0.4)
                    )

    # Add mapping annotation
    for (l, slot) in mapping.items():
        if l not in numbers and not l == "space":
            row = row_numbers[slot[0]]
            column = int(slot[1:3])
            level = slot[4:]

            if level == "":
                pos_x = pos_normal_x
                pos_y = pos_normal_y
                ha = 'left'
                va = 'bottom'
            if level == "Shift":
                pos_x = pos_shift_x
                pos_y = pos_shift_y
                ha = 'left'
                va = 'top'
            if level == "Alt":
                pos_x = pos_alt_x
                pos_y = pos_alt_y
                ha = 'right'
                va = 'bottom'
            if level == "Alt_Shift":
                pos_x = pos_alt_shift_x
                pos_y = pos_alt_shift_y
                ha = 'right'
                va = 'top'
            x = (column * key_width) + column * column_distance + pos_x - row_shift[slot[0]]
            y = (row * key_height) + row * row_distance + pos_y

            if u"d" in l and len(l) > 1:  # dead key
                l = l[0]
                c = dead_color
            else:
                c = special_color

            # use tex for those three characters that won't print
            if l == u"≃":
                l = r"$\simeq$"
            if l == u"‑":
                l = r"$-$"
            if l == "˵":
                l = r" ̏ "
            ax.text(x, y, l,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    fontsize=11,
                    fontweight=special_font,
                    color=c
                    )
    scenario, char_set = get_scenario_and_char_set()

    t = scenario + "\n" + char_set + "\n"
    if not (a == -1 or e == -1 or f == -1 or p == -1):
        t = t + \
            "corpus weights: \nformal - %.2f" % corpus_weights["formal"] + \
            "\ntwitter - %.2f" % corpus_weights["twitter"] + \
            "\ncode - %.2f" % corpus_weights["code"] + "\n\n"

    ax.text(-6, 22, t,
            fontsize=10,
            color='k'
            )
    title = ""
    if not objective == -1:
        # print objective values
        title += "Objective value: %.3f" % objective
    if not (a == -1 or e == -1 or f == -1 or p == -1):
        title = title + " \n Performance: %.2f * %.3f \n Association: %.2f * %.3f \n Familiarity: %.2f * %.3f \n\
        Ergonomics: %.2f * %.3f " % (w_p, p, w_a, a, w_f, f, w_e, e)
    ax.set_title(title, x=0.8, horizontalalignment="right")
    ax.set_xlim([-8, 58])
    ax.set_ylim([-1, 26])
    plt.axis('off')
    if not plotname == "" and not len(axes) > 1:
        fig.savefig(plotname, dpi=300, bbox_inches='tight')

    if not len(axes) > 1:
        return fig, ax, objective, p, a, f, e
    else:
        return ax, objective, p, a, f, e


def compare_intersection(mapping_path, compare_path, w_p=1, w_a=1, w_f=1, w_e=1, axes=[], subplot=[-1, -1, -1, -1]):
    """
    Compares two mappings only with regard to the characters that are present in both mappings. 
    """
    set_scenario_files("scenarioFINAL", "setFINAL")
    corpus_weights = {'formal': 0.7, 'twitter': 0.15, 'code': 0.15}

    mapping = get_mapping(mapping_path)
    compare = get_mapping(compare_path)

    m_ch = set(mapping.keys())
    c_ch = set(compare.keys())

    intersection = m_ch.intersection(c_ch)

    new_mapping = {c: mapping[c] for c in intersection}
    new_compare = {c: compare[c] for c in intersection}

    plot_mapping(new_mapping, "", corpus_weights=corpus_weights, w_p=w_p, w_a=w_a, w_f=w_f, w_e=w_e, axes=axes,
                 subplot=subplot[0])
    plot_mapping_comparison(new_mapping, new_compare, corpus_weights, w_p, w_a, w_f, w_e, _axes=axes,
                            subplot=subplot[1:])
