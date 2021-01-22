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
import codecs
import pandas as pd
import unicodedata
import numpy as np
import glob
import os
import objectives

PYTHONIOENCODING = "utf-8"

VERBOSE = False
_keyslots_file = 'input/keyslots/keyslots.txt'
_fixed_file = ""
_letter_file = 'input/letters/letters.txt'
_character_file = ""
_azerty_file = "input/layouts/azerty.txt"
_similarity_file = 'input/similarity/similarity.txt'
_distance_file_0 = "input/distance/distance0.txt"
_distance_file_1 = "input/distance/distance1.txt"
_distance_file_consistency = "input/distance/distance_consistency.txt"
_frequency_letter_file = ""
_frequency_bigram_file = ""
_ergonomics_file = "input/ergonomics/ergonomics.txt"
_performance_file = "input/performance/performance.txt"
scenario = ""
char_set = ""
_replace_characters = "input/resources/replace_characters.txt"  # characters that should be replaced in frequencies or associations


def get_coherence_pairs():
    """ These are pairs that should be placed next to each other """
    return [(u"(", u")"), (u"[", u"]"), (u"{", u"}"), (u"<", u">"), (u"“", u"”"), (u"‘", u"’"), (u"«", u"»"),
            (u"‹", u"›")]


def get_scenario_and_char_set():
    return scenario, char_set


def set_scenario_files(scenario_name, character_set):
    """
    This function sets the input filenames depending on the given scenario and character set. 
    
    The scenario_name defines the fixed characters. 
    The characer_set defined the to-be-mapped characters and corresponding letter and bigram frequency files. 
    There can be several corpora which should have the same character_set name in it.
    
    As a scenario - character set combination we can create different scenarios that use the same characterset but fix different ones.
    """
    global scenario
    scenario = scenario_name

    global char_set
    char_set = character_set

    global _fixed_file
    _fixed_file = 'input/fixed/fixed_' + scenario + '.txt'

    global _character_file
    _character_file = 'input/characters/characters_' + character_set + '.txt'

    # get all files with that scenario as a list!
    os.chdir("input/frequencies")
    global _frequency_letter_file
    _frequency_letter_file = ["input/frequencies/" + f for f in
                              glob.glob("frequency_letters_*_" + character_set + '.txt')]

    global _frequency_bigram_file
    _frequency_bigram_file = ["input/frequencies/" + f for f in
                              glob.glob("frequency_bigrams_*_" + character_set + '.txt')]
    os.chdir("../..")


# Note diacritics are reresented with a "d" character afterwards
# the unicode keycodes for the diacritic characters, just to be sure the decomposition recognizes them correctly
unicode_diacritic = {u"\u0302": u"^d",
                     u"\u0308": u"¨d",
                     u"\u0303": u"~d",
                     u"\u0306": u"˘d",
                     u"\u030c": u"ˇd",
                     u"\u0311": u"̑d",
                     u"\u0300": u"ˋd",
                     u"\u0301": u"´d",
                     u"\u0304": u"ˉd",
                     u"\u0331": u"_d",
                     u"\u0307": u"˙d",
                     u"\u0323": u".d",
                     u"\u030a": u"°d",
                     u"\u030b": u"˝d",
                     u"\u030f": u"˵d",
                     u"\u0327": u"¸d",
                     u"\u0328": u"˛d",
                     u"\u0326": u",d",
                     u"\u0335": u"-d",
                     u"\u0337": u"/d",
                     u"\u0338": u"/d"
                     }


def get_unicode_diacritic(u):
    if u in unicode_diacritic.keys():
        return unicode_diacritic[u]
    else:
        return u


diacritic_unicode = {u"^": u"\u0302", u"ˆ": u"\u0302",
                     u"¨": u"\u0308", u'"': u"\u0308",
                     u"~": u"\u0303", u"˜": u"\u0303",
                     u"˘": u"\u0306",
                     u"ˇ": u"\u030c",
                     u"̑": u"\u0311",
                     u"ˋ": u"\u0300", u"`": u"\u0300",
                     u"ˊ": u"\u0301", u"´": u"\u0301", u"'": u"\u0301",
                     u"ˉ": u"\u0304",
                     u"_": u"\u0331", u"̲": u"\u0331",
                     u"˙": u"\u0307",
                     u".": u"\u0323",
                     u"°": u"\u030a",
                     u"˝": u"\u030b",
                     u"˵": u"\u030f", u"̏": u"\u030f",
                     u"¸": u"\u0327",
                     u"˛": u"\u0328",
                     u",": u"\u0326", u"̦": u"\u0326",
                     u"-": u"\u0335", u"̵": u"\u0335", u"̶": u"\u0335",
                     u"/": u"\u0337", u"̷": u"\u0337",
                     }


def get_azerty():
    """
        Returns the Azerty keyboard in form of a dict from characters to keyslots (=mapping)
    """
    azerty = pd.read_csv(_azerty_file, index_col=1, sep="\t", encoding='utf-8', quoting=3)
    # azerty.index=azerty.index.str.strip()
    azerty = azerty.to_dict()["keyslot"]
    azerty = {(correct_diacritic(c.strip())): s for c, s in azerty.items()}
    return azerty


def get_characters():
    """
        Returns a list of to-be-mapped characters
    """
    with codecs.open(_character_file, encoding='utf-8') as f:
        characters_file = f.read().splitlines()
    characters = [correct_diacritic(c.strip()) for c in characters_file]
    characters.sort()
    return characters

def get_replace_characters():
    replace = pd.read_csv(_replace_characters, sep="\t", index_col=0,encoding='utf-8', quoting=3)
    replace = replace.to_dict()["mapping_characters"]
    replace = {(correct_diacritic(c.strip())):s for c,s in replace.items()}
    return replace

def correct_diacritic(c):
    """
    Checks if the given character represents a diacritic mark and if so turns it into the common representation of the diacritic mark as 
    given by the unicode_diacritic dict defined above. Otherwise returns the characters as is
    """
    new_c = c
    if len(c) == 2 and "d" in c:  # diacritic
        if c[0] in unicode_diacritic:
            new_c = unicode_diacritic[c[0]]
        else:
            new_c = unicode_diacritic[
                diacritic_unicode[c[0]]]  # correct to a simple representation of the diacritic mark
    return new_c


def get_letters():
    """
        Returns a list of letters in relation to which character mapping should be optimized
    """
    with codecs.open(_letter_file, encoding='utf-8') as f:
        letters_file = f.read().splitlines()
    letters = [c.strip() for c in letters_file]
    letters.sort()
    return letters


def get_fixed_mapping():
    """
        Returns a mapping (dict) of characters that are fixed on the keyboard. Their corresponding slots cannot be filled. 
        They are only considered in the association cost
    """
    fixed = pd.read_csv(_fixed_file, index_col=1, sep="\t", encoding='utf-8', quoting=3)
    # fixed.index=fixed.index.str.strip()
    fixed = fixed.to_dict()["keyslot"]
    fixed = {(correct_diacritic(c.strip())): s for c, s in fixed.items()}
    return fixed


def get_keyslots():
    """
        Returns the list of keyslots that can be mapped to (including those with fixed characters)
    """
    with codecs.open(_keyslots_file, encoding='utf-8') as f:
        keyslots_file = f.read().splitlines()
    keyslots = [c.strip() for c in keyslots_file]

    keyslots.sort()
    return keyslots


def get_character_similarities():
    """
        Returns a dictionary of character tuples to similarity values between the two characters
        Reads the corresponding similarity matrix but removes the characters and letters not given in the corresponding lists
        The dictionary only contains similarity values for those pairs whose similarity is defined to be >0
    """
    similarity_c_c, similarity_c_l = _read_similarity_matrix(_similarity_file)
    return similarity_c_c


def get_character_letter_similarities():
    """
        Returns a dictionary of character-letter tuples to similarity values between each character and each letter
        Reads the corresponding similarity matrix but removes the characters and letters not given in the corresponding lists
        The dictionary only contains similarity values for those pairs whose similarity is defined to be >0
    """
    similarity_c_c, similarity_c_l = _read_similarity_matrix(_similarity_file)
    return similarity_c_l


def get_distances():
    """
        Returns a dictionary of key tuples to distance values between the keys. 
        Returns two such dictionaries, one where the distance is based on the row and column distance, 
        one where it also includes the distance due to different levels (Shift, Alt etc.) The additional level cost is given as input.
    """
    # Distance values (key,key)->d
    distance_level_0 = _read_tuple_list_to_dict(_distance_file_0)
    distance_level_1 = _read_tuple_list_to_dict(_distance_file_1)
    return distance_level_0, distance_level_1

def get_distance_consistency():
    return _read_tuple_list_to_dict(_distance_file_consistency)

def get_probabilities(corpus_weights={}):
    """
        Reads in the frequency file whose filename must be set above. If corpus_weights are given and there are multiple frequency files, 
        the frequency of a letter/bigram is computed as the weighted sum of the frequencies in each corpus. The corpus weights are given 
        as a dictionary of corpus name to weight. The corpus name must be part of the filename, otherwise it is not recognized as such.
        If there are several frequency files defined and no weights are given, each corpus is weighted with the same weight.
        Weights must add up to 1.0
    """
    p_single = {}
    p_bigrams = {}
    if len(_frequency_letter_file) == 1:
        print("only one corpus")
        # only one file, read in and output probabilities
        p_single = pd.read_csv(_frequency_letter_file[0], sep=" ", encoding="utf-8", index_col=0, quoting=3)
        p_single = p_single.to_dict()[u'frequency']

        p_bigrams = _read_tuple_list_to_dict(_frequency_bigram_file[0])
    else:
        if len(corpus_weights) == len(_frequency_letter_file):
            # check if weights sum up to 1:
            if np.sum(list(corpus_weights.values())) != 1:
                raise ValueError('Corpus weights must add up to 1')

            # weight by given weights
            for l_file, b_file in zip(_frequency_letter_file, _frequency_bigram_file):
                single = pd.read_csv(l_file, sep=" ", encoding="utf-8", index_col=0, quoting=3)
                single = single.to_dict()[u'frequency']
                # weight according to given weight:
                weight = -1
                for k, v in corpus_weights.items():
                    if k in l_file:
                        weight = v
                        break;
                if weight == -1:
                    raise ValueError('no weight found for file: %s' % l_file)

                if len(p_single) == 0:
                    p_single = {c: v * weight for c, v in single.items()}
                else:
                    p_single = {c: ((v * weight) + p_single[c]) for c, v in single.items()}

                weight = -1
                # the same for the bigrams
                bigrams = _read_tuple_list_to_dict(b_file)
                for k, v in corpus_weights.items():
                    if k in b_file:
                        weight = v
                        break;
                if weight == -1:
                    raise ValueError('no weight found for file: %s' % b_file)
                if len(p_bigrams) == 0:
                    p_bigrams = {c: v * weight for c, v in bigrams.items()}
                else:
                    p_bigrams = {c: ((v * weight) + p_bigrams[c]) for c, v in bigrams.items()}
        else:
            # weigh everything evenly
            weight = 1 / float(len(_frequency_letter_file))
            for l_file, b_file in zip(_frequency_letter_file, _frequency_bigram_file):
                print("no weights given, weighting each corpus evenly")
                single = pd.read_csv(l_file, sep=" ", encoding="utf-8", index_col=0, quoting=3)
                single = single.to_dict()[u'frequency']
                if len(p_single) == 0:
                    p_single = {c: v * weight for c, v in single.items()}
                else:
                    p_single = {c: ((v * weight) + p_single[c]) for c, v in single.items()}

                bigrams = _read_tuple_list_to_dict(b_file)
                if len(p_bigrams) == 0:
                    p_bigrams = {c: v * weight for c, v in bigrams.items()}
                else:
                    p_bigrams = {c: ((v * weight) + p_bigrams[c]) for c, v in bigrams.items()}

    return p_single, p_bigrams


def get_ergonomics(normalize=1):
    """
        Returns a dictionary with letter-character tuples to ergonomic values
    """
    ergonomics = _read_tuple_list_to_dict(_ergonomics_file)
    if normalize:
        return normalize_dict_values(ergonomics)
    else:
        return ergonomics


def get_performance(normalize=1):
    """
        Returns a dictionary with letter-character tuples to performance values
    """
    # Performance: (key, letter)->t
    performance = _read_tuple_list_to_dict(_performance_file)
    if normalize:
        return normalize_dict_values(performance)
    else:
        return performance


def get_horizontal_distance(slot1, slot2):
    row_numbers = {u"A": 0, u"B": 1, u"C": 2, u"D": 3, u"E": 4}
    return row_numbers[slot1[0]] - row_numbers[slot2[0]]


def get_vertical_distance(slot1, slot2):
    return int(slot1[1:3]) - int(slot2[1:3])


def get_modifier_distance(slot1, slot2):
    level_cost = {
        ["Shift", "Shift"]: 0,
        ["Shift", "Alt"]: 1,
        ["Shift", ""]: 1,
        ["Shift", "Alt_Shift"]: 1,
        ["Alt", "Shift"]: 1,
        ["Alt", "Alt"]: 0,
        ["Alt", ""]: 1,
        ["Alt", "Alt_Shift"]: 1,
        ["", "Shift"]: 1,
        ["", "Alt"]: 1,
        ["", ""]: 0,
        ["", "Alt_Shift"]: 2,
        ["Alt_Shift", "Shift"]: 1,
        ["Alt_Shift", "Alt"]: 1,
        ["Alt_Shift", ""]: 2,
        ["Alt_Shift", "Alt_Shift"]: 0,
    }
    return level_cost[(slot1[4:], slot2[4:])]


def _read_similarity_matrix(path):
    """
        Reads the given file into a dictionary of the form (c1,c2)->similarity
        Filters out all characters that are not in the given character list
        distributes them to two sets, character-character and character-letter similarity
        Already normalized
    """
    characters = get_characters()
    all_chars = get_characters() + get_letters() + list(get_fixed_mapping().keys())
    fixed = get_letters() + list(get_fixed_mapping().keys())
    fixed = [c for c in fixed if not c in characters]  # only those fixed characters that are not in characterlist

    with codecs.open(path, 'r', encoding="utf-8") as similarity_file:
        all_lines = similarity_file.readlines()
        lines = [l.rstrip() for l in all_lines]
        # create dict
        similarity_c_c = {}
        similarity_c_l = {}
        for l in lines:
            parts = l.split(" ")
            if parts[0] != "" and parts[1] != "":
                c1 = correct_diacritic(parts[0])
                c2 = correct_diacritic(parts[1])
                if c1 in all_chars and c2 in all_chars:
                    if c1 in characters and c2 in characters:
                        similarity_c_c[(c1, c2)] = float(parts[5])
                    else:
                        similarity_c_l[(c1, c2)] = float(parts[5])
                else:
                    if VERBOSE:
                        print("%s or %s not in characterset -- association ignored" % (c1, c2))

    return similarity_c_c, similarity_c_l


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


def get_all_input_values(corpus_weights):
    """
    Returns all relevant input values
    """
    azerty = get_azerty()
    letters = get_letters()
    characters = get_characters()
    keyslots = get_keyslots()

    similarity_c_c = get_character_similarities()
    similarity_c_l = get_character_letter_similarities()

    distance_level_0, distance_level_1 = get_distances()

    # read in single probabilities

    p_single, p_bigram = get_probabilities(corpus_weights)
    ergonomics = get_ergonomics()
    performance = get_performance()

    return azerty, \
           characters, \
           keyslots, \
           letters, \
           p_single, p_bigram, \
           performance, \
           similarity_c_c, similarity_c_l, \
           distance_level_0, distance_level_1, \
           ergonomics


def normalize_dict_values(d):
    """
    Normalizes all values to be between 0 and 1 
    """
    # nonzeros = len([v for v in d.values() if not v == 0])
    maximum = np.max(list(d.values()))
    minimum = np.min(list(d.values()))
    sum_all = np.sum(list(d.values()))
    new_dict = {}
    for k, v in d.items():
        new_dict[k] = (v - minimum) / float(maximum - minimum)
        # d[k] = d[k] / float(nonzeros)
        # d[k] = v / float(sum_all)
    return new_dict


def create_normalized_frequency_files(scenario, character_set, letter_files, bigram_files):
    """
    Filters the given frequency files with respect to the given character set and normalizes the frequencies such
    that they sum up to 1.

    Takes care of combined characters and distributing the frequencies accordingly. Then writes them to a file for later use.
    Both contain the frequencies of both, letters and characters (+ fixed characters),
    as well as letter-character and character-character pairs
    The letters and characters that have no frequency available, get zero frequency.
    At the end writes the frequencies to the corresponding file in the input folder.


    Parameters:
        scenario: the scenario to obtain the fixed characters
        char_set: the name of the character set to filter for
        letter_files: a list of  file names that should be filtered for letter frequencies
        bigram_files: a list of  file names that should be filtered for bigram frequencies
    """

    if scenario != "" and character_set != "":
        set_scenario_files(scenario, character_set)

    # character, fixed character, letters
    all_chars = get_characters() + get_letters() + list(get_fixed_mapping().keys())

    # characters that should be replaced by others in the frequency table because they represent the same character
    # according to our understanding
    replace = get_replace_characters()

    # 1. read the frequencies from the corresponding files as they are
    for letter_file in letter_files:
        name_addition = "_"+letter_file.split("_")[-1][:-4]
        p_single_all = pd.read_csv(letter_file, sep=" ", encoding="utf-8", index_col=0, quoting=3)
        p_single_all = p_single_all.dropna()
        p_single_all = p_single_all.to_dict()[u'frequency']
        p_single = {c: 0 for c in all_chars}

        # -------------- SINGLE ------------------------
        # Go through all symbols in the given frequency list.
        # If the symbol is in the character list, add its frequency to the one in the probability list. If not check if it is a
        # composed character. If yes and the diacritic mark is in the character list, add its frequency to the frequency
        # of the corresponding characters (character + diacritic mark)
        for c, v in p_single_all.items():
            if c in replace:
                c = replace[c]

            c = correct_diacritic(c)  # in case this is a wrong form of a "d" annotated diacritic
            if c in p_single.keys():
                p_single[c] += v
            else:
                c_dec = decompose(c)
                if len(c_dec) > 1:
                    # it's a composed character, take it apart and add the frequency to the frequency of its components
                    if c_dec[1] in unicode_diacritic.keys():
                        diacritic = unicode_diacritic[c_dec[1]]

                        if diacritic in p_single.keys():
                            p_single[diacritic] += v
                    letter = c_dec[0]
                    if letter in p_single.keys():
                        p_single[letter] += v
                elif c in diacritic_unicode.keys():
                    # it's a single form of the diacritic mark, e.g. ~. Add frequency to diacritic and to space but only if
                    # it's not available in the characterset
                    if not c in all_chars:
                        diacritic = unicode_diacritic[diacritic_unicode[c]]
                        if diacritic in all_chars:
                            p_single[diacritic] += v
                            p_single["space"] += v
                            # else it's a character we don't care about

        # Check if any of the values remained 0:
        for c, v in p_single.items():
            if v == 0:
                print(u"No frequency for %s" % c)

        # Normalize again
        s = np.sum(list(p_single.values()))
        p_single_normalized = {c: v / float(s) for c, v in p_single.items()}

        # Write LETTERS  to file
        f = codecs.open("input/frequencies/frequency_letters" + name_addition + "_" + character_set + ".txt", 'w',
                        encoding="utf-8")
        f.write("letter frequency\n")
        for c, v in p_single_normalized.items():
            f.write("%s %s" % (c, repr(v)))
            f.write("\n")
        f.close()

    # -------------- BIGRAM ------------------------
    # 3. go through bigrams and correct them according to the given characters,
    # that is letter pairs with accented characters need to be distributed to other letter pairs accoridng to the
    # keypresses that needed to be made.
    for bigram_file in bigram_files:
        name_addition = "_"+bigram_file.split("_")[-1][:-4]
        p_bigrams_all = _read_tuple_list_to_dict(bigram_file)
        p_bigrams = {(c1, c2): 0 for c1 in all_chars for c2 in all_chars}

        for (c1, c2), v in p_bigrams_all.items():
            c1 = correct_diacritic(c1)
            c2 = correct_diacritic(c2)
            if c1 in replace:
                c1 = replace[c1]
            if c2 in replace:
                c2 = replace[c2]

            # decompose first character
            c_c1 = []
            if len(c1) == 0:
                print(c1, c2, v)
            if c1 in all_chars:
                # to-be-mapped character
                c_c1 = [c1]
            elif c1[0] in diacritic_unicode.keys():
                # single form of the diacritic mark, e.g. ~, which needs to be produced by the bigram ~ + space
                diacritic = unicode_diacritic[diacritic_unicode[c1[0]]]  # get the right diacritic
                if diacritic in all_chars:
                    c_c1 = [diacritic, "space"]
            else:
                c1_d = decompose(c1)
                if len(c1_d) > 1:
                    # composed character, otherwise its a character we don't care about
                    c1_1 = get_unicode_diacritic(c1_d[1])  # decompose give diacritic and letter in wrong order
                    c1_2 = get_unicode_diacritic(c1_d[0])
                    if c1_1 in all_chars and c1_2 in all_chars:
                        c_c1 = [c1_1, c1_2]

            # decompose second character
            c_c2 = []
            if c2 in all_chars:
                # to-be-mapped character
                c_c2 = [c2]
            elif c2[0] in diacritic_unicode.keys():
                # single form of the diacritic mark, e.g. ~, which needs to be produced by the bigram ~ + space
                diacritic = unicode_diacritic[diacritic_unicode[c2[0]]]  # get the right diacritic
                if diacritic in all_chars:
                    c_c2 = [diacritic, "space"]
            else:
                c2_d = decompose(c2)
                if len(c2_d) > 1:
                    # composed character, otherwise its a character we don't care about
                    c2_1 = get_unicode_diacritic(c2_d[1])  # decompose give diacritic and letter in wrong order
                    c2_2 = get_unicode_diacritic(c2_d[0])
                    if c2_1 in all_chars and c2_2 in all_chars:
                        c_c2 = [c2_1, c2_2]

            # now add the frequency to the corresponding bigrams
            # c_c1 and c_c2 can contain 0, 1, or 2 letters. If 0 ignore the bigram.
            if len(c_c1) > 0 and len(c_c2) > 0:
                # add bigram for decomposed c1
                if len(c_c1) == 2:
                    p_bigrams[(c_c1[0], c_c1[1])] += v
                # add bigram for decomposed c2
                if len(c_c2) == 2:
                    p_bigrams[(c_c2[0], c_c2[1])] += v
                # add bigram for transition from c1 to c2
                p_bigrams[(c_c1[-1], c_c2[0])] += v


        # normalize
        s = np.sum(list(p_bigrams.values()))
        p_bigrams_normalized = {(c1, c2): v / float(s) for (c1, c2), v in p_bigrams.items()}

        # Write BIGRAMS  to file
        f = codecs.open("input/frequencies/frequency_bigrams" + name_addition + "_" + character_set + ".txt", 'w',
                        encoding="utf-8")
        for (c1, c2), v in p_bigrams_normalized.items():
            f.write("%s %s %s" % (c1, c2, repr(float(v))))
            f.write("\n")
        f.close()


def decompose(c):
    try:
        # print c
        c_d = unicodedata.normalize('NFKD', c)

        return c_d
    except:
        return c