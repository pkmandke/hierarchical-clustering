'''
Trigger script
'''


from pre_process import Doc2vec_wrapper, extract_mapped_doc2vecs
from agglo_clus import Agglo_clus as ag    
import gensim

import time
from datetime import timedelta

def main():
    t1 = time.monotonic()

    model = gensim.models.doc2vec.Doc2Vec.load('../obj/doc2vec/abstracts_etd_doc2vec_5000_docs')
    
    doc_vectors, keys = extract_mapped_doc2vecs(model)
    
    ag_obj = ag(doc_vectors, doc_names = keys, num_clus = 10, linkage='ward', affinity='euclidean', iter_='1')

    ag_obj.clusterize()
    ag_obj.save(name='abstracts_etd_doc2vec_5000_docs_ag_clus.sav')

    print("Time taken {}s".format(timedelta(time.monotonic() - t1)))

main()
