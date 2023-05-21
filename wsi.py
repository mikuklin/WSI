from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer, logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import sys

def load_data():
    train_data = pd.read_csv(
    'https://raw.githubusercontent.com/nlpub/russe-wsi-kit/master/data/main/bts-rnc/train.csv', sep='\t')
    target_words = train_data['word'].unique()
    for word in target_words:
        data = train_data[train_data['word'] == word]
        positions = data['positions'].apply(lambda x: (int(x.split('-')[0]), int(x.split('-')[1]) + 1)).values
        texts = data['context'].values.tolist()
        gold_sense_id = data['gold_sense_id'].values
        yield word, positions, texts, gold_sense_id

def find_model_tokenizer(checkpoint):

    #returns model and tokenizer for checkpoint

    model = AutoModel.from_pretrained(checkpoint, output_hidden_states = True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer

def findword(tokenizer, text, target_word_positions): 

    #finds subword positions of the target word
    #returns the first and the last position

    mapping = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
    begin, end = target_word_positions
    first_map = last_map = 0
    for pos, m in enumerate(mapping):
        if m[1] == end:
            last_map = pos
            break
    for dif, m in enumerate(mapping[last_map::-1]):
        if m[0] == begin:
            first_map = last_map - dif
            break
    return first_map, last_map

def get_embedding(model, tokenizer, texts_batch, target_word_positions, num_layers, layer_pool, subword_pool, device, k=None):

    #returns embedding of the target word

    words = [findword(tokenizer, snt, twp) for snt, twp in zip(texts_batch, target_word_positions)]
    input = tokenizer(texts_batch, padding=True, truncation=True, return_tensors="pt")
    input = input.to(device)
    output = model(**input)
    poolers = {'mean':lambda x: x.mean(1),
               'concat':lambda x: x.flatten(1, 2),
               'max':lambda x: x.max(1)[0],
               'first':lambda x: x[:, 0, :]}
    if layer_pool != 'single':
        layer_pool = poolers[layer_pool]
        hidden_states = torch.stack(output.hidden_states[-num_layers:]).detach()
    else:
        hidden_states = torch.unsqueeze(output.hidden_states[-num_layers], 0)
    if subword_pool == 'special':
        sw_pooled = torch.stack([poolers['first'](hidden_states[:, i, w[0] : w[1] + 1, :]) + \
                                  k * poolers['mean'](hidden_states[:, i, w[0] : w[1] + 1, :]) for i,w in enumerate(words)])
    else:
        subword_pool = poolers[subword_pool]
        sw_pooled = torch.stack([subword_pool(hidden_states[:, i, w[0] : w[1] + 1, :]) for i,w in enumerate(words)])
    if layer_pool != 'single':
        words_emb = layer_pool(sw_pooled).detach().cpu().numpy()
    else:
        words_emb = poolers['mean'](sw_pooled).detach().cpu().numpy()
    torch.cuda.empty_cache()
    return words_emb

def cluster(embeddings, n = None):
    if not n:
        n = 2
        m_score = -1
        for i in range(2, 9):
            clustering = AgglomerativeClustering(i)
            clustering.fit(embeddings)
            score = silhouette_score(embeddings, clustering.labels_)
            if score > m_score:
                m_score = score
                n = i
    clustering = AgglomerativeClustering(n, metric='euclidean',linkage='ward')
    clustering.fit(embeddings)
    return clustering.labels_

def plot_gram_bias(target_words, list_gs, list_vectors): #[['ветка', .., 'ветку'], ..,  ['вид', ..,'вида']]
    senses = []
    words = []
    dists = []
    for (target_word, clusters, vectors) in tqdm(zip(target_words, list_gs, list_vectors)):
        for i, word1 in enumerate(target_word):
            for j, word2 in enumerate(target_word):
                if i < j:
                    dist = np.linalg.norm(vectors[i] - vectors[j])
                    dists.append(dist)
                    if word1 == word2 and clusters[i] == clusters[j]:
                        senses.append('same')
                        words.append('same')
                    if word1 != word2 and clusters[i] == clusters[j]:
                        senses.append('same')
                        words.append('diff')
                    if word1 == word2 and clusters[i] != clusters[j]:
                        senses.append('diff')
                        words.append('same')
                    if word1 != word2 and clusters[i] != clusters[j]:
                        senses.append('diff')
                        words.append('diff')
    data = pd.DataFrame({'sense':senses, 'words':words, 'distance':dists})
    sns.set(font_scale=2)
    g = sns.displot(data, x = 'distance', col="sense", row="words", facet_kws=dict(sharey=True, sharex=True), height=10, aspect=2)
    def specs(x, **kwargs):
        plt.axvline(x.mean(), c='k', ls='-', lw=2.5)
        plt.axvline(x.median(), c='orange', ls='--', lw=2.5)
        mean = mpatches.Patch(color='k', label='mean')
        med = mpatches.Patch(color='orange', label='med')
        plt.legend(handles=[mean, med])
    g.map(specs, "distance")

def gen(lst1, lst2, n):
    for i in range(0, len(lst1), n):
        yield lst1[i: i + n], lst2[i: i + n]

def wsi(checkpoint, num_layers, layer_pool, subword_pool, k=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device: {}'.format(device))
    model, tokenizer = find_model_tokenizer(checkpoint)
    model = model.to(device)
    batch_size = 20
    aris = np.array([])
    count = np.array([])
    target_words = []
    list_clusters = []
    list_vectors = []
    list_gs = []
    for word, positions, texts, gold_sense_id in load_data():
        words = []
        for text, position in zip(texts, positions):
            words.append(text[position[0]: position[1]])
        target_words.append(words)
        embs = np.array([None])
        for batch, pos in gen(texts, positions, batch_size):
            new_embs =  get_embedding(
                model, tokenizer, batch, pos, num_layers, layer_pool, subword_pool, device, k)
            embs = np.r_[embs, new_embs] if embs.any() else new_embs
        clusters = cluster(embs, 3) # n is fixed
        #clusters = cluster(embs, np.unique(gold_sense_id).size) # n by golden labels
        #clusters = cluster(embs) # n by silhouette score
        list_clusters.append(clusters)
        list_vectors.append(embs)
        list_gs.append(gold_sense_id)
        ari = adjusted_rand_score(gold_sense_id, clusters)
        print("ari on word '{}': {:f}".format(word, ari))
        count = np.append(count, len(texts))
        aris = np.append(aris, ari * count[-1])
    print('average ari: {}'.format(np.sum(aris) / np.sum(count)))
    return target_words, list_gs, list_vectors

if __name__ == "__main__":
    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    lemmas, list_gs, list_vectors = wsi('xlm-roberta-large', 5, 'single', 'special', 0.4)
    plot_gram_bias(lemmas, list_gs, list_vectors)
