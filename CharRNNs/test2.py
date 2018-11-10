import numpy as np
import os
import codecs

def main():
    model_path = os.path.join('model', "AAAAtest")
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

def pick_top_n(preds, vocab_size, top_n=3):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    p = list(p)
    print(p)
    c = np.random.choice(vocab_size, 1, replace=False, p=p)
    return c[0]
if __name__ == '__main__':
    preds = [0, 0, 0.1, 0.2, 0.2, 0.5, 0]
    a = pick_top_n(preds=preds, vocab_size=["a", "b", "c", "d", "e", "f", "g"])
    print(a)