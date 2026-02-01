import re
import networkx as nx
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from typing import List, Tuple


def modify_graph(original_graph):
	"""
	트리플의 각 토큰을 소문자/공백 정리
	"""
	modified_graph = []
	for x in original_graph:
		modified_graph.append([str(t).lower().strip() for t in x])
	return modified_graph


def get_tokens(gold_edges, pred_edges):
	nlp = English()
	tokenizer = Tokenizer(nlp.vocab, infix_finditer=re.compile(r'''[;]''').finditer)

	gold_tokens = []
	pred_tokens = []

	for i in range(len(gold_edges)):
		gold_tokens_edges = []
		pred_tokens_edges = []

		for sample in tokenizer.pipe(gold_edges[i]):
			gold_tokens_edges.append([j.text for j in sample])
		for sample in tokenizer.pipe(pred_edges[i]):
			pred_tokens_edges.append([j.text for j in sample])
		gold_tokens.append(gold_tokens_edges)
		pred_tokens.append(pred_tokens_edges)

	return gold_tokens, pred_tokens


def split_to_edges(graphs):
	pgrph = []
	for graph in graphs:
		pgrph.append([";".join(str(triple)).lower().strip() for triple in graph])
	return pgrph


def return_eq_node(node1, node2):
	"""
	노드 동치 판단(라벨 기준)
	"""
	return node1['label'] == node2['label']


def return_eq_edge(edge1, edge2):
	"""
	엣지 동치 판단(라벨 기준)
	"""
	return edge1['label'] == edge2['label']


