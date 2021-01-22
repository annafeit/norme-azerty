#!/usr/bin/env python3.8

###############################################################################
# Copyright (c) 2019, Maximilian John, Andreas Karrenbauer and Anna Maria Feit
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     # Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     # Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     # The name of the author may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

def name(prefix, l, j):
    return str(prefix) + "(" + str(l) + "," + str(j) + ")"


def printX(n):
    output = []
    for k in range(n):
        line = ""
        for i in range(n):
            line += name("x", k, i) + " "
        output.append("\n" + line)
    return output


def printAssignment(n):
    output = []
    for k in range(n):
        line = ""
        for i in range(n):
            line += " + " + name("x", k, i)
        output.append("\n" + line + " = 1")

    for i in range(n):
        line = ""
        for k in range(n):
            line += " + " + name("x", k, i)
        output.append("\n" + line + " = 1")
    return output


def fixKey(i, k):
    print(name("x", i, k) + " = 1")


def disallowKey(i, k):
    print(name("x", i, k) + " = 0")


def setEqual(i, j, k, l):
    print(name("x", i, k) + " - " + name("x", j, l) + " = 0")


def setTripleConstraint(i1, i2, i3, k1, k2, k3):
    print(name("x", i1, k1) + " - " + name("x", i2, k2) + " - " + name("x", i3, k3) + " = 0")


def printKaufmanBroeckxInequalities(probabilities, distances):
    output = []
    n = len(probabilities)

    c = [[0 for i in range(n)] for j in range(n)]
    for k in range(n):
        for i in range(n):
            for l in range(n):
                for j in range(n):
                    c[k][i] += probabilities[k][l] * distances[i][j];

    for k in range(n):
        for i in range(n):
            output.append("\n" + str(c[k][i]) + " " + name("x", k, i))
            for l in range(n):
                for j in range(n):
                    pd = probabilities[k][l] * distances[i][j]
                    if pd > 0.0000001:
                        output.append("\n" + " + " + str(pd) + " " + name("x", l, j))
            output.append("\n" + " - " + name("w", k, i) + " <= " + str(c[k][i]))
    return output


def printKaufmanBroeckxObjective(n, linearcosts):
    output = []
    line = ""
    for k in range(n):
        for i in range(n):
            line += " + " + str(linearcosts[k][i]) + " " + name("x", k, i)
    output.append("\n" + line)

    for k in range(n):
        line = ""
        for i in range(n):
            line += " + " + name("w", k, i)
        output.append("\n" + line)
    return output


def generateModel(n, probabilities, distances, linearcosts):
    output = ["minimize"]
    output += printKaufmanBroeckxObjective(n, linearcosts)

    output.append("\nsubject to")
    output += printAssignment(n)

    output += printKaufmanBroeckxInequalities(probabilities, distances)

    # Here you could add the special constraints
    # fixKey(1,2)
    # disallowKey(3,3)
    # setEqual(1,3,2,4)
    # setTripleConstraint(1,2,1,3,2,4)

    output.append("\nbinaries")
    output += printX(n)
    output.append("\nend")
    return output


def kaufmannbroeckx_reformulation(input_filename, output_filename):
    with open(input_filename, "r") as file:
        # read all lines and strip off newline char
        lines = file.read().splitlines()

        # we ignore two lines of comments, third line is the number of letters and keys
        n = int(lines[2])
        # ignore next comment, the following lines are the probabilities
        line_counter = 4
        probabilities = []
        for i in range(n):
            # Expecting n lines
            l = lines[i + line_counter]
            if "#" in l:
                raise ValueError("fewer probabilities than number of letters indicated")
            else:
                row = l.split(" ")
                row = [float(v) for v in row]
                probabilities.append(row)
        # ignore next comment #next lines are the distances
        line_counter += n + 1
        distances = []
        for i in range(n):
            # Expecting n lines
            l = lines[i + line_counter]
            if "#" in l:
                raise ValueError("fewer distances than number of letters indicated")
            else:
                row = l.split(" ")
                row = [float(v) for v in row]
                distances.append(row)
        # ignore the next five lines, this is some legacy stuff we don't use here...
        line_counter += n + 5
        # next lines are the linear costs
        linearcosts = []
        for i in range(n):
            # Expecting n lines
            l = lines[i + line_counter]
            if "#" in l:
                raise ValueError("fewer linear costs than number of letters indicated")
            else:
                row = l.split(" ")
                row = [float(v) for v in row]
                linearcosts.append(row)

        # generate lines for printing
        output = generateModel(n, probabilities, distances, linearcosts)
        # print lines to file
        with open(output_filename, "w") as output_file:
            output_file.writelines(output)
