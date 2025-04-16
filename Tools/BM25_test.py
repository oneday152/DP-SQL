import math
from collections import Counter
from Levenshtein import distance as levenshtein_distance

class BM25Fuzzy:
    def __init__(self, documents, k1=1.5, b=0.75, delta=1.0):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.avg_doc_len = sum(len(doc) for doc in documents) / len(documents)
        self.doc_freqs = self._calculate_doc_freqs()
        self.idf = self._calculate_idf()

    def _calculate_doc_freqs(self):
        doc_freqs = {}
        for doc in self.documents:
            for word in set(doc):
                if word not in doc_freqs:
                    doc_freqs[word] = 0
                doc_freqs[word] += 1
        return doc_freqs

    def _calculate_idf(self):
        idf = {}
        for word, freq in self.doc_freqs.items():
            idf[word] = math.log((len(self.documents) - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def _fuzzy_match_score(self, query_word, doc_word):
        dist = levenshtein_distance(query_word, doc_word)
        max_len = max(len(query_word), len(doc_word))
        return 1 - (dist / max_len)

    def score(self, query):
        scores = []
        for doc in self.documents:
            score = 0.0 
            doc_len = len(doc)
            doc_counter = Counter(doc)
            for query_word in query:
                max_fuzzy_score = 0.0 
                best_matching_word = None
                
                for doc_word in doc:
                    fuzzy_score = self._fuzzy_match_score(query_word, doc_word)
                    if fuzzy_score > max_fuzzy_score:
                        max_fuzzy_score = fuzzy_score
                        best_matching_word = doc_word
                
                if max_fuzzy_score > 0:
                    term_freq = doc_counter[best_matching_word]
                    idf = self.idf.get(best_matching_word, 0)
                    score += (idf * 
                            ((term_freq * (self.k1 + 1)) / 
                             (term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))) * 
                            max_fuzzy_score)
            scores.append(score)
        return scores

