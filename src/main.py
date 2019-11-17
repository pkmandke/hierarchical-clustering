'''
Trigger script
'''


from pre_process import Doc2vec_wrapper

def main():
    
    doc2vec_model = Doc2vec_wrapper(json_path='../data/30Kmetadata.json', n_docs=5000)
    
    doc2vec_model.generate_tokens()
    
    doc2vec_model.load_model_and_build_vocab(vector_size=128, epochs=20, workers=5)
    
    doc2vec_model.train()
    
    doc2vec_model.save_model()
    
main()
