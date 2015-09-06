#!/usr/bin/env python
import dtree
import monkdata
import random
import matplotlib.pyplot as plot

sets = [monkdata.monk1, monkdata.monk2, monkdata.monk3]

entropies = [dtree.entropy(s) for s in sets]

def printlines(values):
    for line in values:
       print(', '.join(map(str, line)))

print("Initial entropies:")
print(entropies)
print("")


gain = [[dtree.averageGain(s, attr) for attr in monkdata.attributes] for s in sets]

print("Expected gain:")
printlines(gain)
print("")

def tests(pair):
    tree=dtree.buildTree(pair[0], monkdata.attributes)
    return [
            pair[2],
            dtree.check(tree,pair[0]),
            dtree.check(tree,pair[1])
    ]


setpairs = [
        [monkdata.monk1, monkdata.monk1test, "MONK-1"],
        [monkdata.monk2, monkdata.monk2test, "MONK-2"],
        [monkdata.monk3, monkdata.monk3test, "MONK-3"],
        ]

checks = [tests(pair) for pair in setpairs]
print("Check training vs test data:")
printlines(checks)
print("")

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint= int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def unzip(values):
    return [list(t) for t in zip(*values)]

fractions = [0.3,0.4,0.5,0.6,0.7,0.8]
series=[]
for pair in setpairs:
    values = []
    for fraction in fractions:
        s = pair[0]
        testdata = pair[1]
        training, validation = partition(s, fraction)
        tree=dtree.buildTree(training, monkdata.attributes)
        keepPruning = True
        while keepPruning:
            alternatives = dtree.allPruned(tree)
            keepPruning = False
            for alternative in alternatives:
                if(dtree.check(alternative,validation) > dtree.check(tree,validation)):
                    tree = alternative
                    keepPruning = True
        error=dtree.check(tree,testdata)
        values.append((fraction,error))
    #convert pairs to two lists [xs, ys]
    data=unzip(values)
    data.append(pair[2])
    series.append(data)

print("Pruned trees:")
printlines(series)
print("")

lines=[]
for s in series:
    xs = s[0]
    ys = s[1]
    label=s[2]
    lines += plot.plot(xs, ys,label=label)
plot.legend(handles=lines)
plot.show()
