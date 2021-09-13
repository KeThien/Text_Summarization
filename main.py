from Summarizer import Summarizer
from absl import app, flags
import numpy as np
import re


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "text_path",
    "texts/to_summarize.txt",
    "path to the text to summarize"
)

flags.DEFINE_string(
    "model",
    "flaubert",
    "Use either 'camembert' or 'flaubert' for model"
)

flags.DEFINE_string(
    "method",
    "mean",
    "mean / Clustering or Graph summarization"
)

flags.DEFINE_integer(
    "nb_sentences",
    5,
    "number of sentences in the summary"
)

flags.DEFINE_bool(
    "visualization",
    False,
    "visualize text or not"
)


def summarize(model, method, text, nb_sentences, viz=False):
    summarizer = Summarizer(model, log=True)

    summarizer.fit(text)
    
    if method == 'mean':
        summary = summarizer.mean_similarity_summary(nb_sentences=nb_sentences)

    elif method == 'clustering':
        summary, cluster_results = summarizer.clustering_summary(nb_clusters=nb_sentences, nb_top=2, return_clusters=True)
        labels, cluster_indices = cluster_results
        if viz:
            summarizer.text_visualization(cluster_labels=labels, plot_lib='plotly')

    elif method == 'graph':
        summary = summarizer.graph_summary(nb_sentences=nb_sentences)

    return summary


def load_preprocess_text(path):
    with open(path, 'r') as f:
        text = f.read()

    r = '\(.+\)'
    print('numbre de caractères du texte :', len(text))
    text = re.sub(r, '', text)
    l = len(text)
    text = text.split('.')

    return np.array(text), l


def main(_):
    model = FLAGS.model
    method = FLAGS.method 
    path_text = FLAGS.text_path
    nb_sentences = FLAGS.nb_sentences
    viz = False

    text, long = load_preprocess_text(path_text)
    print("nom du modèle: ", model)
    summary = summarize(model, method, text, nb_sentences, viz)
    result = ". ".join(summary) + '.'
    print("longueur du texte nettoyé: ", long)
    print("longueur du résumé: ", len(result))
    print("résumé:\n", result)


if __name__ == '__main__':
    app.run(main)
