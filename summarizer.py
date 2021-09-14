from transformers import pipeline
import os


def summarize(text, model="plguillou/t5-base-fr-sum-cnndm", device=-1, min_length=75, max_length=300, truncation=True):
    '''
    This function takes a text and returns a summarized version of it.

    :param text: text to summarize
    :param model: model to use (default: plguillou/t5-base-fr-sum-cnndm)
    :param device: device to use (default: -1 = cpu)
    :param min_length: minimum length of the summary (default: 75)
    :param max_length: maximum length of the summary (default: 300)
    :param truncation: truncation of the summary (default: True)
    :return summarized version object of the text
    '''
    summarizer = pipeline("summarization", model=model,
                          tokenizer=model, framework="pt", device=device)
    summarized = summarizer(text, min_length=min_length,
                            max_length=max_length, truncation=truncation)
    return summarized


if __name__ == "__main__":
    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = dirpath + "/../texts/to_summarize.txt"

# Open and read the article
    f = open(filepath, "r", encoding="utf-8")
    to_tokenize = f.read()

    print(summarize(to_tokenize))
