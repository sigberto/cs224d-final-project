import numpy as np

def load_dataset(f1, f2, f3, batch_size, max_paragraph_len, in_batches, random):
    fd1, fd2, fd3 = open(f1), open(f2), open(f3)
    batch1 = []
    batch2 = []
    batch3 = []
    num_lines = sum(1 for _ in fd1)
    fd1.seek(0)
    if not random:
        for i in xrange(num_lines-1):
            line1, line2, line3 = fd1.readline(), fd2.readline(), fd3.readline()
            context = [int(x) for x in line1.split()]
            if len(context) > max_paragraph_len:
                continue
            batch1.append(context)

            batch2.append([int(x) for x in line2.split()])
            batch3.append([int(x) for x in line3.split()])

            if in_batches and (len(batch1) == batch_size or i == num_lines-2):
                yield batch1, batch2, batch3
                batch1 = []
                batch2 = []
                batch3 = []
            elif not in_batches and i == num_lines-2:
                yield batch1, batch2, batch3
    else:
        all_inds = range(num_lines)
        indx = np.random.choice(all_inds, size=batch_size, replace=False)
        all1 = fd1.readlines()
        all2 = fd2.readlines()
        all3 = fd3.readlines()
        for i in indx:
            batch1.append(all1[i])
            batch2.append(all2[i])
            batch3.append(all3[i])
        yield batch1, batch2, batch3



# PYTHON GENERATOR 
# yield is like a return, it returns a value and keeps going on . 


# PYTHON GENERATOR
# yield is like a return, it returns a value and keeps going on .
def load_single_dataset(f1, f2, f3, batch_size):
    fd1, fd2, fd3 = open(f1), open(f2), open(f3)
    batch1 = []
    batch2 = []
    batch3 = []
    fd1.seek(0)
    for i in xrange(batch_size):
        line1, line2, line3 = fd1.readline(), fd2.readline(), fd3.readline()
        batch1.append([int(x) for x in line1.split()])
        batch2.append([int(x) for x in line2.split()])
        batch3.append([int(x) for x in line3.split()])

    return batch1, batch2, batch3

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


def read_dataset(paragraph_file, question_file, answer_file, max_paragraph_len):
    paragraph_stream, question_stream, answer_stream = open(paragraph_file), open(question_file), open(answer_file)
    dataset = []

    while True:
        raw_paragraph = [int(x) for x in paragraph_stream.readline().split()]
        if not raw_paragraph: break
        if len(raw_paragraph) > max_paragraph_len: continue

        raw_question = [int(x) for x in question_stream.readline().split()]
        #if len(raw_question.split(" ")) <= 2: continue # toss out bad questions
        raw_answer = [int(x) for x in answer_stream.readline().split()]
        
        dataset.append((raw_paragraph, raw_question, raw_answer))

    question_stream.close()
    paragraph_stream.close()
    answer_stream.close()

    return dataset


def get_sample_dataset(dataset, sample=100):
    size = len(dataset)
    random_indexes = np.random.choice(size, sample, replace=False)
    output = []

    for idx in random_indexes:
        output.append(dataset[idx])

    return output