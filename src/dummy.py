import linecache
path = '/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/train-v1/offenseval-training-v1.tsv'
maxi = 0

with open(path, "r") as f:
    for line in f:
        contents = line.split("\t")
        l = len(contents[1].split(" ")[1:])
        print(l)
        if(l > maxi):
            maxi = l
print("\n",maxi)
