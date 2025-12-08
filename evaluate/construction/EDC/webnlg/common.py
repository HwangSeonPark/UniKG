import os
import regex as re
import itertools
import statistics
import xml.etree.ElementTree as ET
from typing import Tuple

from bs4 import BeautifulSoup
import nltk
from nltk.util import ngrams
import string
from unidecode import unidecode
import ast
from nervaluate import Evaluator

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def _try_indent(tree):
    try:
        ET.indent(tree, space="\t", level=0)
    except AttributeError:
        pass


def convert_to_xml(result_path: str, gold_path: str, max_length_diff=None) -> Tuple[str, str]:
    normalized_path = os.path.normpath(os.path.abspath(result_path))
    parent_dir = os.path.basename(os.path.dirname(normalized_path))
    if parent_dir:
        output_dir = parent_dir
    else:
        output_dir = os.path.splitext(os.path.basename(normalized_path))[0] or "output"
    os.makedirs(f"./result_xmls/{output_dir}", exist_ok=True)
    pred_xml_path = os.path.join(f"./result_xmls/{output_dir}", f"pred.xml")
    ref_xml_path = os.path.join(f"./result_xmls/{output_dir}", f"ref.xml")

    pred_triplets = [l.strip() for l in open(result_path, "r").readlines()]
    gold_triplets = [l.strip() for l in open(gold_path, "r").readlines()]

    collected_pred_triplets = []
    collected_gold_triplets = []

    for idx, triplets in enumerate(pred_triplets):
        try:
            evaled_triplets = ast.literal_eval(triplets)
            for triplet in evaled_triplets:
                if len(triplet) != 3:
                    raise Exception
                for element in triplet:
                    if not isinstance(element, str):
                        raise Exception
            collected_pred_triplets.append(evaled_triplets)
            collected_gold_triplets.append(ast.literal_eval(gold_triplets[idx]))
        except Exception:
            pass

    assert len(collected_pred_triplets) == len(collected_gold_triplets)

    pred_root_node = ET.Element("benchmark")
    pred_entries_node = ET.SubElement(pred_root_node, "entries")

    gold_root_node = ET.Element("benchmark")
    gold_entries_node = ET.SubElement(gold_root_node, "entries")

    for idx in range(len(collected_gold_triplets)):
        length_diff = abs(len(collected_gold_triplets[idx]) - len(collected_pred_triplets[idx]))
        if max_length_diff is not None and length_diff > int(max_length_diff):
            continue

        pred_entry_node = ET.SubElement(pred_entries_node, "entry")
        pred_generated_tripleset = ET.SubElement(pred_entry_node, "generatedtripleset")
        for triplet in collected_pred_triplets[idx]:
            gtriplet_node = ET.SubElement(pred_generated_tripleset, "gtriple")
            gtriplet_node.text = f"{triplet[0]} | {triplet[1]} | {triplet[2]}"

        gold_entry_node = ET.SubElement(gold_entries_node, "entry")
        gold_reference_tripleset = ET.SubElement(gold_entry_node, "modifiedtripleset")
        for triplet in collected_gold_triplets[idx]:
            rtriplet_node = ET.SubElement(gold_reference_tripleset, "mtriple")
            rtriplet_node.text = f"{triplet[0]} | {triplet[1]} | {triplet[2]}"

    pred_tree = ET.ElementTree(pred_root_node)
    _try_indent(pred_tree)
    pred_tree.write(pred_xml_path)

    gold_tree = ET.ElementTree(gold_root_node)
    _try_indent(gold_tree)
    gold_tree.write(ref_xml_path)

    return pred_xml_path, ref_xml_path


def getRefs(filepath):
    with open(filepath, encoding="utf-8") as fp:
        refssoup = BeautifulSoup(fp, "lxml")

    refsentries = refssoup.find("benchmark").find("entries").find_all("entry")

    allreftriples = []
    for entry in refsentries:
        entryreftriples = []
        modtriplesref = entry.find("modifiedtripleset").find_all("mtriple")
        for modtriple in modtriplesref:
            entryreftriples.append(modtriple.text)
        allreftriples.append(entryreftriples)

    newreflist = []
    eidx = 0  # 빈 문자열을 고유 토큰으로 치환하기 위한 인덱스
    for entry in allreftriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r"_", " ", newtriple).lower()
            newtriple = re.sub(r"\s+", " ", newtriple).lower()
            newtriple = unidecode(newtriple)
            adjusttriple = newtriple.split(" | ")
            manualmodified = re.search(r"^(.*?)(\s\((.*?)\))$", adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
            # 빈 문자열 오답 처리: 각 요소가 빈칸이면 고유 토큰으로 치환
            for i in range(len(adjusttriple)):
                if not adjusttriple[i].strip():
                    adjusttriple[i] = f'<emp_{eidx}>'
                    eidx += 1
            newtriple = " | ".join(adjusttriple)
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return allreftriples, newreflist


def getCands(filepath):
    with open(filepath, encoding="utf-8") as fp:
        candssoup = BeautifulSoup(fp, "lxml")

    candssentries = candssoup.find("benchmark").find("entries").find_all("entry")

    allcandtriples = []
    for entry in candssentries:
        entrycandtriples = []
        modtriplescand = entry.find("generatedtripleset").find_all("gtriple")
        for modtriple in modtriplescand:
            entrycandtriples.append(modtriple.text)
        allcandtriples.append(entrycandtriples)

    newcandlist = []
    eidx = 0  # 빈 문자열을 고유 토큰으로 치환하기 위한 인덱스
    for entry in allcandtriples:
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r"_", " ", newtriple).lower()
            newtriple = re.sub(r"\s+", " ", newtriple).lower()
            newtriple = unidecode(newtriple)
            adjusttriple = newtriple.split(" | ")
            manualmodified = re.search(r"^(.*?)(\s\((.*?)\))$", adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
            # 빈 문자열 오답 처리: 각 요소가 빈칸이면 고유 토큰으로 치환
            for i in range(len(adjusttriple)):
                if not adjusttriple[i].strip():
                    adjusttriple[i] = f'<emp_{eidx}>'
                    eidx += 1
            newtriple = " | ".join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcandtriples, newcandlist


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1


def nonrefwords(newreflist, newcandlist, foundnum, ngramlength):
    while ngramlength > 0:
        ngramlist = list(ngrams(newcandlist, ngramlength))
        for ngram in ngramlist:
            if find_sub_list(list(ngram), newreflist) is not None:
                findnewref = find_sub_list(list(ngram), newreflist)
                newrefindex = list(range(findnewref[0], findnewref[1] + 1))
                for idx in newrefindex:
                    newreflist[idx] = "FOUNDREF-" + str(foundnum) + "-" + str(idx)
                findnewcand = find_sub_list(list(ngram), newcandlist)
                newcandindex = list(range(findnewcand[0], findnewcand[1] + 1))
                for idx, val in enumerate(newcandindex):
                    newcandlist[val] = "FOUNDCAND-" + str(foundnum) + "-" + str(newrefindex[idx])
                foundnum += 1
                newreflist, newcandlist = nonrefwords(newreflist, newcandlist, foundnum, ngramlength)
                break
        ngramlength -= 1
    return newreflist, newcandlist


def getrefdict(newreflist, newcandlist, tripletyperef, tripletypecand, baseidx):
    try:
        firstfoundidx = newcandlist.index([i for i in newcandlist if re.findall(r"^FOUNDCAND", i)][0])
        candidatefound = "y"
    except IndexError:
        candidatefound = "n"

    if candidatefound == "y":
        unlinkedlist = []
        beforelist = []
        afterlist = []

        if newcandlist[firstfoundidx].endswith("-0"):
            beforelinked = "y"
            firstcand = re.search(r"^(FOUNDCAND-\d+)-", newcandlist[firstfoundidx]).group(1)
        else:
            beforelinked = "n"

        lastfoundidx = None
        afterlinked = None
        if (newreflist[-1].startswith("FOUNDREF")) and (not newcandlist[-1].startswith("FOUNDCAND")):
            lastfound = [i for i in newcandlist if re.findall(r"^FOUNDCAND", i)][-1]
            candversion = newreflist[-1].replace("FOUNDREF", "FOUNDCAND")
            if lastfound == candversion:
                lastfoundidx = newcandlist.index([i for i in newcandlist if re.findall(r"^FOUNDCAND", i)][-1])
                afterlinked = "y"
                lastcand = re.search(r"^(FOUNDCAND-\d+)-", lastfound).group(1)

        unlinknumber = 1
        for idx, can in enumerate(newcandlist):
            if not can.startswith("FOUNDCAND"):
                if (idx < firstfoundidx) and (beforelinked == "y"):
                    newcandlist[idx] = firstcand + "-LINKED"
                    beforelist.append(firstcand + "-LINKED")
                elif (lastfoundidx != None) and (afterlinked != None) and (idx > lastfoundidx) and (afterlinked == "y"):
                    newcandlist[idx] = lastcand + "-LINKED"
                    afterlist.append(lastcand + "-LINKED")
                else:
                    unlinkedlist.append("NOTFOUND-" + str(unlinknumber))
            else:
                unlinknumber += 1

        totallist = beforelist + newreflist + afterlist + unlinkedlist

        refstart = len(beforelist)
        refend = (len(beforelist) + len(newreflist)) - 1
        refdictlist = [{"label": tripletyperef, "start": baseidx + refstart, "end": baseidx + refend}]

        totallist2 = [x.replace("FOUNDREF", "FOUNDCAND") for x in totallist]

        canddictlist = []
        currentcandidate = ""
        beginidx = ""
        endidx = ""
        collecting = "n"
        for idx, candidate in enumerate(totallist2):
            if (candidate.startswith("FOUNDCAND")) or (candidate.startswith("NOTFOUND")):
                collecting = "y"
                curcan = re.search(r"^((.*?)-\d+)", candidate).group(1)
                if curcan != currentcandidate:
                    if currentcandidate != "":
                        endidx = idx - 1
                        canddictlist.append(
                            {"label": tripletypecand, "start": baseidx + beginidx, "end": baseidx + endidx}
                        )
                    currentcandidate = curcan
                    beginidx = idx

                if idx == len(totallist2) - 1:
                    endidx = idx
                    canddictlist.append({"label": tripletypecand, "start": baseidx + beginidx, "end": baseidx + endidx})
            else:
                if collecting == "y":
                    endidx = idx - 1
                    canddictlist.append({"label": tripletypecand, "start": baseidx + beginidx, "end": baseidx + endidx})

    else:
        if len(newreflist) == 0:
            refdictlist = []
            canddictlist = [{"label": tripletypecand, "start": baseidx, "end": baseidx + (len(newcandlist) - 1)}]
            totallist = newcandlist
        elif len(newcandlist) == 0:
            canddictlist = []
            refdictlist = [{"label": tripletyperef, "start": baseidx, "end": baseidx + (len(newreflist) - 1)}]
            totallist = refdictlist
        else:
            totallist = newreflist + newcandlist
            refdictlist = [{"label": tripletyperef, "start": baseidx, "end": baseidx + (len(newreflist) - 1)}]
            canddictlist = [
                {"label": tripletypecand, "start": baseidx + len(newreflist), "end": baseidx + (len(totallist) - 1)}
            ]

    return candidatefound, refdictlist, canddictlist, totallist


def evaluaterefcand(reference, candidate):
    newreference = reference.split(" | ")
    newcandidate = candidate.split(" | ")

    if (len(newreference) > 1) and (len(newcandidate) > 1):
        indextriple = newreference
    elif len(newreference) == 1:
        indextriple = newcandidate
        newreference = ["", "", ""]
    else:
        indextriple = newreference
        newcandidate = ["", "", ""]

    subjectreflist = None
    subjectcandlist = None
    subjecttotallist = None
    predicatereflist = None
    predicatecandlist = None
    predicatetotallist = None
    objectreflist = None
    objectcandlist = None
    objecttotallist = None
    subjectfound = ""
    predicatefound = ""
    objectfound = ""

    for idx, attrib in enumerate(indextriple):
        refsub = newreference[idx]
        candsub = newcandidate[idx]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r"^[" + re.escape(string.punctuation) + r"]+$", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"^[" + re.escape(string.punctuation) + r"]$", x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        if idx == 0:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, "SUB", "SUB", 0)
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
        elif idx == 1:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "PRED", "PRED", len(subjecttotallist)
            )
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
        else:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "OBJ", "OBJ", len(subjecttotallist) + len(predicatetotallist)
            )
            objectfound = candidatefound
            objectreflist = refdictlist.copy()
            objectcandlist = canddictlist.copy()
            objecttotallist = totallist.copy()

    switchmatchfound = "n"
    if (subjectfound == "n") and (objectfound == "n"):
        refsub = newreference[0]
        candsub = newcandidate[2]
        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)
        reflist = [x.lower() for x in reflist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, "SUB", "OBJ", 0)

        refsub = newreference[2]
        candsub = newcandidate[0]
        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)
        reflist = [x.lower() for x in reflist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(
            newreflist, newcandlist, "OBJ", "SUB", len(totallist) + len(predicatetotallist)
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()

            candidatefound, refdictlist, canddictlist, totallist = getrefdict(
                newreflist, newcandlist, "PRED", "PRED", len(subjecttotallist)
            )
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
            switchmatchfound = "y"
        else:
            switchmatchfound = "n"

    if ((subjectfound == "n") and (predicatefound == "n")) and (switchmatchfound == "n"):
        refsub = newreference[0]
        candsub = newcandidate[1]
        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)
        reflist = [x.lower() for x in reflist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, "SUB", "PRED", 0)

        refsub = newreference[1]
        candsub = newcandidate[0]
        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)
        reflist = [x.lower() for x in reflist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(
            newreflist, newcandlist, "PRED", "SUB", len(totallist)
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            predicatefound = candidatefound2
            predicatereflist = refdictlist2.copy()
            predicatecandlist = canddictlist2.copy()
            predicatetotallist = totallist2.copy()
            switchmatchfound = "y"
        else:
            switchmatchfound = "n"

    if ((predicatefound == "n") and (objectfound == "n")) and (switchmatchfound == "n"):
        refsub = newreference[1]
        candsub = newcandidate[2]
        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)
        reflist = [x.lower() for x in reflist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound, refdictlist, canddictlist, totallist = getrefdict(
            newreflist, newcandlist, "PRED", "OBJ", len(subjecttotallist)
        )

        refsub = newreference[2]
        candsub = newcandidate[1]
        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)
        reflist = [x.lower() for x in reflist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        candlist = [x.lower() for x in candlist if re.search(r"[" + re.escape(string.punctuation) + r"]", x) == None]
        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(
            newreflist, newcandlist, "OBJ", "PRED", len(subjecttotallist) + len(totallist)
        )

        if (candidatefound == "y") or (candidatefound2 == "y"):
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()
            switchmatchfound = "y"
        else:
            switchmatchfound = "n"

    allrefdict = subjectreflist + predicatereflist + objectreflist
    allcanddict = subjectcandlist + predicatecandlist + objectcandlist
    evaluator = Evaluator([allrefdict], [allcanddict], tags=["SUB", "PRED", "OBJ"])

    eval_out = evaluator.evaluate()
    if isinstance(eval_out, tuple):
        results = eval_out[0]
        results_per_tag = eval_out[1]
    else:
        results = eval_out
        results_per_tag = {}

    return results, results_per_tag


def calculateAllScores(newreflist, newcandlist):
    totalsemevallist = []
    totalsemevallistpertag = []

    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) != len(newreflist[idx]):
            differencebetween = abs(len(newcandlist[idx]) - len(newreflist[idx]))
            differencelist = [""] * differencebetween
            if len(newcandlist[idx]) < len(newreflist[idx]):
                newcandlist[idx] = newcandlist[idx] + differencelist
            else:
                newreflist[idx] = newreflist[idx] + differencelist

    for idx, candidate in enumerate(newcandlist):
        candidatesemeval = []
        candidatesemevalpertag = []
        for triple in candidate:
            triplesemeval = []
            triplesemevalpertag = []
            for reference in newreflist[idx]:
                results, results_per_tag = evaluaterefcand(reference, triple)
                triplesemeval.append(results)
                triplesemevalpertag.append(results_per_tag)
            candidatesemeval.append(triplesemeval)
            candidatesemevalpertag.append(triplesemevalpertag)
        totalsemevallist.append(candidatesemeval)
        totalsemevallistpertag.append(candidatesemevalpertag)

    return totalsemevallist, totalsemevallistpertag


def calculateSystemScore(totalsemevallist, totalsemevallistpertag, newreflist, newcandlist):
    selectedsemevallist = []
    selectedsemevallistpertag = []
    selectedalignment = []
    selectedscores = []

    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) > len(newreflist[idx]):
            choosecands = list(
                itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx])], len(totalsemevallist[idx][0]))
            )
            choosecands = set([tuple(sorted(i)) for i in choosecands])
            choosecands = list(map(list, choosecands))
        else:
            choosecands = [list(range(len(newcandlist[idx])))]

        if len(newcandlist[idx]) > len(newreflist[idx]):
            choosescore = list(
                itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newreflist[idx]))
            )
            choosescore = [list(x) for x in choosescore]
        else:
            choosescore = list(
                itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newcandlist[idx]))
            )
            choosescore = [list(x) for x in choosescore]

        combilist = list(itertools.product(choosecands, choosescore))
        totaldict = {"totalscore": 0}

        for combination in combilist:
            combiscore = 0
            zipcombi = list(zip(combination[0], combination[1]))
            collectedsemeval = []
            collectedsemevalpertag = []

            for zc_idx, zc in enumerate(zipcombi):
                collectedscores = totalsemevallist[idx][zc[0]][zc[1]]
                f1score = statistics.mean(
                    [
                        collectedscores["ent_type"]["f1"],
                        collectedscores["partial"]["f1"],
                        collectedscores["strict"]["f1"],
                        collectedscores["exact"]["f1"],
                    ]
                )
                combiscore += f1score
                collectedsemeval.append(collectedscores)
                assert combination[0][zc_idx] == zc[0] and combination[1][zc_idx] == zc[1]
                collectedsemevalpertag.append(totalsemevallistpertag[idx][zc[0]][zc[1]])

            if (combiscore > totaldict["totalscore"]) or (len(totaldict) == 1):
                totaldict = {
                    "totalscore": combiscore,
                    "combination": combination,
                    "semevallist": collectedsemeval,
                    "semevalpertaglist": collectedsemevalpertag,
                }

        selectedsemevallist = selectedsemevallist + totaldict["semevallist"]
        selectedsemevallistpertag = selectedsemevallistpertag + totaldict["semevalpertaglist"]
        selectedalignment.append(totaldict["combination"])
        selectedscores.append(totaldict["totalscore"] / len(candidate))

    return selectedsemevallist, selectedsemevallistpertag, selectedalignment, selectedscores


