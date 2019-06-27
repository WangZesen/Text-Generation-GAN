import pickle
import gensim
import numpy as np

def process(model_file, source_file, summary_file, output_emb_file, output_vocab_file, output_doc_file, min_time_source, max_time_source, min_time_summary, max_time_summary):
    """
        Pre-process files to create embedding and documents

        Args:
            model_file: path to word2vec model
            source_file: path to source text file
            summary_file: path to summary file
            output_emb_file: output path for embedding
            output_vocab_file: output path for vocabulary file
            output_doc_file: output path for combined and paried documents
    """

    source = pickle.load(open(source_file, 'rb'))
    summary = pickle.load(open(summary_file, 'rb'))

    doc = []

    filter_cnt = 0

    for i in range(len(source)):
        if (len(source[i]) <= max_time_source) and (len(source[i]) >= min_time_source) and (len(summary[i]) <= max_time_summary) and (len(summary[i]) >= min_time_summary):
            doc.append((source[i], summary[i]))
        else:
            filter_cnt += 1

    print (f'{filter_cnt} samples are filtered out of {len(source)} by length limit')

    
    model = gensim.models.Word2Vec.load(model_file)

    vocab = {'<END>': 0, '<START>': 1, '<NEXT>': 2, '<UNKNOWN>': 3}
    cnt = 4

    model = gensim.models.Word2Vec.load(model_file)

    for idx, key in enumerate(model.wv.vocab):
        if not (key in vocab):
            vocab[key] = cnt
            cnt += 1

    embedding = np.zeros((len(vocab), model.wv['hello'].shape[0]))

    print (f'Vocabulary size: {len(vocab)}')

    for key, idx in vocab.items():
        try: # '<UNKNOWN>' 
            embedding[idx] = model.wv[key]
        except:
            pass

    i = 0
    invalid_cnt = 0

    while i < len(doc):
        invalid = False
        for token in doc[i][1]:
            if (not (token in doc[i][0])) and (not (token in vocab)):
                invalid = True
                break
        if invalid:
            invalid_cnt += 1
            del doc[i]
        else:
            i += 1
    
    print (f'{invalid_cnt} samples are filtered out among {len(doc) + invalid_cnt} by OOV')


    pickle.dump(doc, open(output_doc_file, 'wb'))
    pickle.dump(embedding, open(output_emb_file, 'wb'))
    pickle.dump(vocab, open(output_vocab_file, 'wb'))

if __name__ == '__main__':
    process(model_file = './data/word2vec_sum.model',
        source_file = './data/doc_source.p',
        summary_file = './data/doc_summary.p',
        output_emb_file = './data/emb.p',
        output_doc_file = './data/doc.p',
        output_vocab_file = './data/vocab.p',
        min_time_source = 100, 
        max_time_source = 1000,
        min_time_summary = 10,
        max_time_summary = 120)