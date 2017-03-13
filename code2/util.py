def load_dataset(f1, f2, f3, batch_size, in_batches):
    fd1, fd2, fd3 = open(f1), open(f2), open(f3)
    batch1 = []
    batch2 = []
    batch3 = []
    num_lines = sum(1 for _ in fd1)
    fd1.seek(0)
    for i in xrange(num_lines-1):
        line1, line2, line3 = fd1.readline(), fd2.readline(), fd3.readline()
        batch1.append([int(x) for x in line1.split()])
        batch2.append([int(x) for x in line2.split()])
        batch3.append([int(x) for x in line3.split()])

        if in_batches and (len(batch1) == batch_size or i == num_lines-2):
            yield batch1, batch2, batch3
            batch1 = []
            batch2 = []
            batch3 = []
        elif not in_batches and i == num_lines-2:
            yield batch1, batch2, batch3


# PYTHON GENERATOR 
# yield is like a return, it returns a value and keeps going on . 

