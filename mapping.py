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

# -*- coding: utf-8 -*-
import codecs
import re
from read_input import *
import operator

PYTHONIOENCODING = "utf-8"


def set_scenario_from_file(filename):
    firstline = open(filename).readline()
    if not "scenario" in firstline:
        print("No scenario and set defined in file %s" % filename)
    else:
        parts = firstline.split(",")
        scenario = parts[0].split("=")[1]
        char_set = parts[1].split("=")[1]

        print("setting scenario %s, set %s" % (scenario, char_set))
        set_scenario_files(scenario, char_set)

        w_p = float(parts[2].split("=")[1])
        w_a = float(parts[3].split("=")[1])
        w_f = float(parts[4].split("=")[1])
        w_e = float(parts[5].split("=")[1])

        corpus_weights = {'formal': float(parts[6].split("=")[1]),
                          'twitter': float(parts[7].split("=")[1]),
                          'code': float(parts[8].split("=")[1])}
    return get_mapping(filename), w_p, w_a, w_f, w_e, corpus_weights


def get_mapping(mapping_name):
    """
        Reads in the given mapping and returns a dictionary of letters to keys. If the given mapping is a dictionary, 
        does nothing an returns the mapping
        mpaping_name can be a path to different file formats
    """
    # read in mapping
    if type(mapping_name) == str:
        if mapping_name.split(".")[-1] == "mst":
            mapping = create_map_from_reformulation(mapping_name)
        elif mapping_name.split(".")[-1] == "txt":
            mapping = create_map_from_txt(mapping_name)
        return mapping
    else:
        return mapping_name


def log_mst(mapping, path):
    """
        Stores the given mapping in an mst file with the given path. Format:
        character key
    """
    mstfile = codecs.open(path, 'w', encoding="utf-8")
    for character, key in mapping.items():
        mstfile.write('%s %s\n' % (character, key))
    mstfile.close()


def log_mapping(mst, path):
    """
        Stores the given mapping in an txt file with the given path. Format:
        character key
    """
    mappingfile = codecs.open(path, 'w', encoding="utf-8")
    if type(mst) == str:
        mapping = get_mapping(mst)
        # write commented lines
        with codecs.open(mst, encoding="utf-8") as f:
            for line in f:
                if line[0] == "#":
                    mappingfile.write(line)
                else:
                    break;
    else:
        mapping = mst
    for character, key in mapping.items():
        mappingfile.write('%s %s\n' % (character, key))
    mappingfile.close()


def create_map_from_txt(path):
    """
    Reads a mapping from a file
    Each line must have the form character - space - key
    lines starting with # are ignored
    """
    mst = codecs.open(path, 'r', encoding="utf-8")
    all_lines = mst.read().splitlines()

    mapping = {}
    for i in range(0, len(all_lines)):
        line = all_lines[i]
        if "#" in line and len(line.split(",")) > 2:
            continue;  # skip comments
        else:
            var_val = line.split(" ")
            mapping[correct_diacritic(var_val[0].strip())] = var_val[1]

    mst.close()
    return mapping


def create_map_from_reformulation(path):
    """ 
        creates the mapping from the refomulated solution .mst file
    """
    scenario, char_set = get_scenario_and_char_set()
    if scenario == "":
        scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights = get_firstline_parameter(path)
        if scenario != -1:
            set_scenario_files(scenario, char_set)
        else:
            print("First define scenario and char_set!")
            return -1

    # read in characters and keyslots
    keyslots = get_keyslots()
    characters = get_characters()

    # read in mst file line by line and create mapping
    mst = codecs.open(path, 'r', encoding="utf-8")

    all_lines = mst.read().splitlines()

    mapping = {}
    for line in all_lines:
        if line[0] != "#":  # not a comment
            var_val = line.split(" ")
            variable = var_val[0]
            # take only "x" decision variables which are set to 1
            if var_val[1] == "1" and variable[0] == "x":
                # decode number
                maps = variable[2:-1].split(",")
                c_number = int(maps[0])
                s_number = int(maps[1])
                # map number to character/keyslot
                if c_number < len(characters):
                    character = characters[c_number]
                    slot = keyslots[s_number]
                    mapping[character] = slot
    return mapping


def get_firstline_parameter(path):
    """
    Parses the parameters logged in the first line of the file
    """
    if path.endswith(".mst"):
        firstline = open(path).readline()
        parts = firstline.split(",")

        scenario = parts[0].split("=")[1]
        char_set = parts[1].split("=")[1]

        w_p = float(parts[2].split("=")[1])
        w_a = float(parts[3].split("=")[1])
        w_f = float(parts[4].split("=")[1])
        w_e = float(parts[5].split("=")[1])

        corpus_weights = {'formal': float(parts[6].split("=")[1]),
                          'twitter': float(parts[7].split("=")[1]),
                          'code': float(parts[8].split("=")[1])}

        return scenario, char_set, w_p, w_a, w_f, w_e, corpus_weights
    else:
        return -1, -1, -1, -1, -1, -1


def change_characters(mapping, change):
    """
    Changes the mapping according to the partial mapping given in "change". 
    Returns an error in the case where a key has two characters mapped on it.
    """
    new_mapping = mapping.copy()
    for c, v in change.items():
        new_mapping[c] = v

    rev_multidict = {}
    for key, value in new_mapping.items():
        rev_multidict.setdefault(value, set()).add(key)
    duplicates = [values for key, values in rev_multidict.items() if len(values) > 1]
    if len(duplicates) > 0:
        print("Error: The following characters have the same key mapped to them:")
        for s in duplicates:
            for c in s:
                print(u"%s:%s" % (c, new_mapping[c]))

        return mapping
    else:
        return new_mapping
