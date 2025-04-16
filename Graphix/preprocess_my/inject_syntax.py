# -*- coding: utf-8 -*-
import os, json, pickle, argparse, sys, time
import pdb
import stanza
from stanza.models.common.doc import Document

from supar import Parser

MOD = ['nn', 'amod', 'advmod', 'rcmod', 'partmod', 'poss', 'neg', 'predet', 'acomp', 'advcl', 'ccomp', 'tmod',
                  'mark', 'xcomp', 'appos', 'npadvmod', 'infmod','nummod','nmod'] + \
                 ['num', 'number', 'quantmod'] + ['pobj', 'dobj', 'iobj']

def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += ["\"", tok[1:-1], "\""]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["\"", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "\"" ]
        elif tok in quotation_marks:
            new_question.append("\"")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += ["\"", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question

def inject_syntax_dataset(processor, dataset, output_path=None):
    syntax_dataset = []
    for idx, data in enumerate(dataset):
        entry = processor.inject_syntax(data)
        syntax_dataset.append(entry)

        if idx % 100 == 0:
            print("************************ processing {}th dataset ************************".format(idx))
    if output_path:
        pickle.dump(syntax_dataset, open(output_path, "wb"))
    return syntax_dataset

def inject_syntax_dataset_json(processor, dataset, mode='train', output_path=None):
    syntax_dataset = []
    for idx, data in enumerate(dataset):
        entry = processor.inject_syntax(data)
        if mode == 'dev':
            entry['graph_idx'] = idx + 8577 
        else:
            entry['graph_idx'] = idx
        syntax_dataset.append(entry)

        if idx % 1000 == 0:
            print("************************ processing {}th dataset ************************".format(idx))
    if output_path:
        json.dump(syntax_dataset, open(output_path, "w"), indent=4)
    return syntax_dataset

class DEP():

    def __init__(self):
        super(DEP, self).__init__()


    def convert_to_pretagged_format(self,entry):
        tokens = entry['raw_question_toks']
        lemma_tokens = entry['processed_question_toks']
        upos_tags = entry['upos_tags']
        xpos_tags = entry['xpos_tags']
        feats_tags = entry['feats_tags']
        if not (len(tokens) == len(lemma_tokens) == len(upos_tags) == len(xpos_tags) == len(feats_tags)):
            raise ValueError("The length of tokens, UPOS, XPOS, and Feats must be the same.")
    
        pretagged_doc = []
        
        for i in range(len(tokens)):
            word_info = {
                'id': i + 1,
                'text': tokens[i],
                'lemma': lemma_tokens[i], 
                'upos': upos_tags[i],
                'xpos': xpos_tags[i],
                'feats': feats_tags[i]
            }
            pretagged_doc.append(word_info)
        
        return [[word_info for word_info in pretagged_doc]]


    def acquire_dep(self, entry):
        dep_dict = {}
        question = ' '.join(quote_normalization(entry['question_toks']))
        parsed_question = self.parser.predict(question, lang='en', prob=False, verbose=False).sentences[0]
        arcs = [i - 1 for i in parsed_question.arcs[0]]
        rels = parsed_question.rels[0]
        for tgt, src in enumerate(arcs):
            rel = rels[tgt]
            if rel in MOD:
                rel = 'mod'
            else:rel = 'arg'
            dep_dict[tgt] = [src, rel]

        return dep_dict

    def inject_syntax(self, entry):
        question = ' '.join(quote_normalization(entry['question_toks']))
        relation_matrix = entry['relations']
        ori_question = entry['processed_question_toks']
        pretagged_doc = self.convert_to_pretagged_format(entry)
        pretagged_doc = Document(pretagged_doc)

        # Inject relations:
        nlp = stanza.Pipeline(lang='en', processors='depparse',depparse_pretagged=True)
        doc = nlp(pretagged_doc)
        arcs = [int(word.head)-1 for sent in doc.sentences for word in sent.words]
        rels = [word.deprel for sent in doc.sentences for word in sent.words]
        print("arcs:", arcs)
        print("rels:", rels)
        # Construct:
        if len(relation_matrix) != len(arcs):
            print("mismatched: {}".format(question))
            print("processed: {}".format(ori_question))
        
        for tgt, src in enumerate(arcs):
            if src < 0:
                continue
            rel = rels[tgt]
            print("Target:", tgt, "Source:", src, "Relation:", rel)  # Inspect relationships
            if rel in MOD:
                relation_matrix[src][tgt] = 'question-question-modifier'
            else:
                relation_matrix[src][tgt] = 'question-question-argument'
        
        entry['relations'] = relation_matrix
        
        # Print matrix with row and column names
        # print("Relation Matrix:")
        # print("\t" + "\t".join(ori_question))  # Print column headers
        # for i, row in enumerate(relation_matrix):
        #     print(ori_question[i], "\t" + "\t".join(row))  # Print row name followed by its values

        # print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

        return entry
