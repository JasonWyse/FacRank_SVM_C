import os
import sys

save_path="./result"
vali_measure=""
dataset=""
algorithm=""
cur_subset=""
ndcg = {}
Map = {}
result = [0]*12
def getFile(dir):
	files = []
	dir = os.path.abspath(dir)
	for i in os.listdir(dir):
		files.append(os.path.join(dir,i))
	return files
def ifNotExistMk(path):
	"""
		check if the fold exists ,if not make it
	"""
	if not os.path.exists(path):
		os.makedirs(path)
def check(a,b,c):
	assert(a == "performance")
	assert(b == "on")
	assert(c == "set")

def doLine(line,lineNum):
	global vali_measure,dataset,algorithm,cur_subset,Map,ndcg
	if lineNum == 1:
		_,vali_measure = map(lambda x: x.strip().lower(),line.split(":"))
	elif lineNum == 2:
		_,dataset = map(lambda x: x.strip().lower(),line.split(":"))
	elif lineNum == 3:
		_,algorithm = map(lambda x: x.strip(),line.split(":"))
	elif lineNum == 4 or lineNum == 17 or lineNum == 30 :
		a,b,cur_subset,c = map(lambda x: x.strip().lower(),line.split(" "))
		check(a,b,c)
		if cur_subset.startswith("test"):
			cur_subset = "testing"
		elif cur_subset.startswith("train"):
			cur_subset = "training"
		elif cur_subset.startswith("vali"):
			cur_subset = "validation"
		else:
			sys.exit(-1)

	elif (lineNum > 5 and lineNum < 11) or \
				(lineNum > 18 and lineNum < 24) or \
				(lineNum > 31 and lineNum < 37):
		line = line.strip("\t")
		x = map(lambda x: x.strip(),line.split("\t"))[1:]
		assert(len(x) == 11)
		x = map(float,x)

		if ndcg.has_key(cur_subset):
			ndcg[cur_subset].append(x)
		else:
			ndcg[cur_subset] = []
			ndcg[cur_subset].append(x)
	elif (lineNum > 11 and lineNum < 17) or \
				(lineNum > 24 and lineNum < 30) or \
				(lineNum > 37 and lineNum < 43):
		line = line.strip("\t")
		x = map(lambda x: x.strip(),line.split("\t"))[11]
		x = float(x)
		if Map.has_key(cur_subset):
			Map[cur_subset].append(x)
		else:
			Map[cur_subset] = []
			Map[cur_subset].append(x)
	elif lineNum < 43:
		return
	else:
		sys.exit(-1)

def average():
	global ndcg,Map,result
	test = ndcg["testing"]
	sum = [0.0]*12
	for tmp in test:
		for i in range(0,len(tmp)):
			sum[i] += tmp[i]
	for tmp in Map["testing"]:
		sum[11] += tmp
	for i in range(0,12):
		result[i] = sum[i]/5.0


def readFile(fileName):
	lineNum = 1
	with open(fileName) as fin:
		for line in fin:
			line = line.strip("\n")
			if line == "":
				continue
			doLine(line,lineNum)
			lineNum += 1
	average()

def save():
	global result
	#print save_path
	path = os.path.join(save_path,vali_measure,dataset)

	ifNotExistMk(path)

	filename = "%s_%s"%(vali_measure,dataset)
	path = os.path.join(path,filename)
	flag = False
	if not os.path.exists(path):
		flag = True
	with open(path,'a') as fout:
		if flag :
			fout.write("Algorithm\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\t"+\
				"NDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10\tMeanNDCG\tMap\n")
			flag = False
		fout.write("%s\t"%algorithm)
		for tmp in result:
			fout.write("%.4f\t"%tmp)
		fout.write("\n")

def run():
	global vali_measure,dataset,algorithm,cur_subset,Map,ndcg,save_path,result
	if len(sys.argv) < 2:
		sys.exit(-1)
	if len(sys.argv) == 3:
		save_path = os.path.abspath(sys.argv[2])
	else:
		save_path = os.path.abspath(save_path)
	files = getFile(sys.argv[1])
	for f in files:
		vali_measure=""
		dataset=""
		algorithm=""
		cur_subset=""
		ndcg = {}
		Map = {}
		result = [0]*12
		readFile(f)

		save()

if __name__ == "__main__":
	run()