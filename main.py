import os
from summarizer import summarize

if __name__ == "__main__":
    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = dirpath + "/../texts/to_summarize.txt"

# Open and read the article
    f = open(filepath, "r", encoding="utf-8")
    to_tokenize = f.read()

    print(summarize(to_tokenize, device=0))
