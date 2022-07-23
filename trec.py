import rank_metric as metrics
import pandas as pd
import numpy as np

class TrecEvaluation:

    #Queries sao os cases, qrels e o path para as relevancias 
    def __init__(self, queries, qrels):
        
        # fields: docid, rel
        self.queries = queries
        #Le-se o ficheiro separando valores com \t (tab) onde cada um dos valores encaixa na matriz names por ordem (case/query, lixo, document_id/clinical_trial_id, relevancia)
        #Fica entao guardado o documento em tabela
        self.relevance_judgments = pd.read_csv(qrels, sep='\t', names=["query_id", "dummy", "docid", "rel"])
       
        #Vao-se buscar os valores dos ids dos documentos/clinical_trials usando a funcao unique que percorre a coluna dos docid
        self.judged_docs = np.unique(self.relevance_judgments['docid'])
       
        #So o numero dos documentos unicos
        self.num_docs = len(self.judged_docs)
        
        
    def all_relevancies(self):
        return self.relevance_judgments
    
    def relevant_docs_out_of_all_relevant(self, result, query_id):
        #Todas as linhas relevantes a query
        aux = self.relevance_judgments.loc[self.relevance_judgments['query_id'] == int(query_id)]
        #linhas cujas relevancias sao diferentes de 0
        rel_docs = aux.loc[aux['rel'] != 0]
        #documentos relevantes
        query_rel_docs = rel_docs['docid']
        #numero de documentos relevantes a query
        total_relevant = query_rel_docs.count()
        #top 100 dos resultados
        top100 = result['_id'][:100]
        #resultados que sao relevantes
        true_pos = np.intersect1d(top100,query_rel_docs)
        #numero de resultados que sao relevantes
        x = np.size(true_pos)
        
        return [x, total_relevant]
    
    def rel_documents(self, query_id):
        return self.relevance_judgments.loc[self.relevance_judgments['query_id'] == int(query_id)]
        
    
    def eval(self, result, query_id):
        
        total_retrieved_docs = result.count()[0]
        
        aux = self.relevance_judgments.loc[self.relevance_judgments['query_id'] == int(query_id)]
        
        rel_docs = aux.loc[aux['rel'] != 0]
        
        query_rel_docs = rel_docs['docid']
        
        relv_judg_list = rel_docs['rel']
        
        total_relevant = relv_judg_list.count()
        
        if total_relevant == 0:
            return [0, 0, 0, 0, 0]
        
        #-------------------------------------------------------------------
        
        top10 = result['_id'][:10]
        top100 = result['_id'][:100]
    
        true_pos = np.intersect1d(top10,query_rel_docs)
        
        p10 = np.size(true_pos) / 10
        
        #-------------------------------------------------------------------
        
        true_pos = np.intersect1d(top100,query_rel_docs)
        
        recall = np.size(true_pos) / total_relevant

        #-------------------------------------------------------------------
        
        # Compute vector of results with corresponding relevance level 
        
        relev_judg_results = np.zeros((total_retrieved_docs,1))       
      
        for index, doc in rel_docs.iterrows():
            z = ((result['_id'] == doc.docid)*doc.rel).to_numpy()
            relev_judg_results = relev_judg_results + z
            
        # Normalized Discount Cummulative Gain
        p10 = metrics.precision_at_k(relev_judg_results[0], 10)
        ndcg10 = metrics.ndcg_at_k(r = relev_judg_results[0], k = 10, method = 1)
        ap = metrics.average_precision(relev_judg_results[0], total_relevant)
        mrr = metrics.mean_reciprocal_rank(relev_judg_results[0])
        
        return [p10, recall, ap, ndcg10, mrr]

    def evalPR(self, scores, query_id):

        #aux = self.relevance_judgments.loc[self.relevance_judgments['topic_turn_id'] == (topic_turn_id)]
        aux = self.relevance_judgments.loc[self.relevance_judgments['query_id'] == int(query_id)]
        idx_rel_docs = aux.loc[aux['rel'] != (0)]
        [dummyA, rank_rel, dummyB] = np.intersect1d(scores['_id'], idx_rel_docs['docid'], return_indices=True)
        rank_rel = np.sort(rank_rel) + 1
        total_relv_ret = rank_rel.shape[0]
        if total_relv_ret == 0:
            return [np.zeros(11, ), [], total_relv_ret]
        recall = np.arange(1, total_relv_ret + 1)
        recall = recall / idx_rel_docs.shape[0]
        precision = np.arange(1, total_relv_ret + 1)
        precision = precision / rank_rel
        precision_interpolated = np.maximum.accumulate(precision) 
        recall_11point = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        precision_11point = np.interp(recall_11point, recall, precision)
       

        if False:
            import matplotlib.pyplot as plt
            print(total_relv_ret)
            print(rank_rel)
            print(recall)
            print(precision)
            plt.plot(recall, precision, color='b', alpha=1)  # Raw precision-recall
            plt.plot(recall, precision_interpolated, color='r', alpha=1)  # Interpolated precision-recall
            plt.plot(recall_11point, precision_11point, color='g', alpha=1)  # 11-point interpolated precision-recall
            plt.show()

        return [precision_11point, recall_11point, total_relv_ret]

