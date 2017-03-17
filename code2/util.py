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
