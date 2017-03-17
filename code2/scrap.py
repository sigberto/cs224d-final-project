import util

output = util.load_dataset("data/squad/train.ids.context", "data/squad/train.ids.question",
                                 "data/squad/train.span", 10, 750, in_batches=True, random=True)
p, q, a = output.next()
print(p)

