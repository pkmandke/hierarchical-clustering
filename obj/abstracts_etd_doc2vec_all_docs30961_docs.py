'''
Trigger script
'''


from pre_process import Doc2vec_wrapper, extract_mapped_doc2vecs
import kmeans
import gensim

import time
from datetime import timedelta

def main():
    t1 = time.monotonic()
    
    doc2vec_model = Doc2vec_wrapper(json_path='../data/30Kmetadata.json', n_docs=-1)

    doc2vec_model.generate_tokens()
    print("Tokens generated in {}s".format(timedelta(seconds=time.monotonic() - t1)))

    doc2vec_model.load_model_and_build_vocab(vector_size=128, min_count=4, dbow_words=0, epochs=15, workers=5)

    doc2vec_model.train()

    doc2vec_model.save_model(path='../obj/doc2vec/abstracts_etd_doc2vec_all_docs')

    print("Time taken {}s".format(timedelta(seconds=time.monotonic() - t1)))

main()
