from typing import Tuple
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from .webnlg.common import convert_to_xml, getRefs, getCands


def calculateExactTripleScore(reflist, candlist) -> Tuple[float, float, float]:
    """
    Exact triple 기반 P/R/F1 
    """
    newreflist = [[string.lower() for string in sublist] for sublist in reflist]
    newcandlist = [[string.lower() for string in sublist] for sublist in candlist]

    allclasses = newcandlist + newreflist
    allclasses = [item for items in allclasses for item in items]
    allclasses = list(set(allclasses))

    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(newcandlist)
    mrbin = lb.fit_transform(newreflist)

    precision = precision_score(mrbin, mcbin, average="macro")
    recall = recall_score(mrbin, mcbin, average="macro")
    f1 = f1_score(mrbin, mcbin, average="macro")
    return float(precision), float(recall), float(f1)


def f1(pred_path: str, gold_path: str) -> Tuple[float, float, float]:
    """
    XML 변환 → 정규화 → Exact Triple 매크로 P/R/F1 계산
    """
    pred_xml_path, gold_xml_path = convert_to_xml(pred_path, gold_path, max_length_diff=None)
    reflist, _ = getRefs(gold_xml_path)
    candlist, _ = getCands(pred_xml_path)
    return calculateExactTripleScore(reflist, candlist)


