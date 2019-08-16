import numpy as np

def rank5_accuracy(preds, labels):
    """

    :param preds:N x T matrix of predictions from model
    :param labels: ground truth labels
    :return: (rank1, rank5) rank1 and 5 accuracies
    """

    rank1 = 0
    rank5 = 0

    for (p, gt) in zip(preds, labels):
        p = np.argsort(p)[::-1]

        if gt in p[:5]:
            rank5 += 1

        if gt == p[0]:
            rank1 += 1

    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    return (rank1, rank5)