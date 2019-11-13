'''
Test script
'''


from pre_process import Doc2vec_wrapper

def main()
    
    doc2vec_model = Doc2vec_wrapper(json_path='/mnt/ceph/cme/30Ketds/30Kmetadata.json', n_docs=2)
    
    doc2vec_model.generate_tokens()
    
    doc2vec_model.load_model_and_build_vocab(vector_size=128, epochs=1)
    
    doc2vec_model.train()
    
    doc2vec_model.save_model()
    
main()
