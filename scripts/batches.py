import math


def generate_batches(batch_length, X, Y):
    current_batch = 0
    batch_num = 0
    samples_count = X.shape[1]
    batches_to_create = math.floor(samples_count / batch_length)
    batches = []

    while batch_num < batches_to_create:
        # section off the next X_batch
        X_batch = X[:, current_batch:current_batch + batch_length]
        # section off the next Y_batch
        Y_batch = Y[:, current_batch:current_batch + batch_length]

        # generate the batch
        batch = (X_batch, Y_batch)
        batches.append(batch)
        current_batch += batch_length
        batch_num += 1

    final_batch = (X[:, current_batch:X.shape[1]], Y[:, current_batch:Y.shape[1]])
    batches.append(final_batch)

    return batches

