# Token ids ranked by how often the model generates them, for the MTP draft head's small vocabulary
# (MOEX_DRAFT_RANK=file MOEX_DRAFT_VOCAB=N). Input: moex --batch stdout (one line of comma-separated ids per prompt).
#   python tools/draft_rank.py gen.txt [more.txt ...] > draft_rank.txt
import collections, sys

c = collections.Counter()
for path in sys.argv[1:]:
    for line in open(path):
        c.update(int(t) for t in line.strip().split(',') if t)
for t, _ in c.most_common():
    print(t)
