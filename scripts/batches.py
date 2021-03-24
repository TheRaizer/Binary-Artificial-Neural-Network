import math


def generate_batches(batch_length, X, Y):
    current_batch = 0
    batch_num = 0
    samples_count = X.shape[1]
    batches_to_create = math.floor(samples_count / batch_length)
    batches = []

    while batch_num < batches_to_create:
        batch = (X[:, current_batch:current_batch + batch_length], Y[:, current_batch:current_batch + batch_length])
        batches.append(batch)
        current_batch += batch_length
        batch_num += 1

    final_batch = (X[:, current_batch:X.shape[1]], Y[:, current_batch:Y.shape[1]])
    batches.append(final_batch)

    return batches

