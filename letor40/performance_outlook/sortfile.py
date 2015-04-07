import re
import sys
import os


def sortFile(filename):
	"""
		@param filename: The file need to be sorted,filename must be abstract path
	"""
	#print filename
	ql = []
	with open(filename,'r') as fin:
		for line in fin:
			line = line.strip('\n')
			if line == "":
				continue
			m=re.match(r'^(\d+) (qid\:)*([^\s]+).*?\#docid = ([^\s]+).*$',line)
			if m:
				label = m.group(1)
				qid = int(m.group(3))
				temp_did={'qid':qid,'label':label,'line':line}
				ql.append(temp_did)
			else:
				print line
				sys.exit(-1)
	#sort
	ql2= sorted(ql, key=lambda d:d['qid'])

	#save
	with open(filename,'w+') as fout:
		for tmp in ql2:
			fout.write(tmp['line'] + '\n')


def sortAll(dir):
	"""
		@dir the root path
	"""
	dir = os.path.abspath(dir)
	#get folds
	for i in os.listdir(dir):
		path = os.path.join(dir,i)
		if os.path.isdir(path):
			#get subset
			for j in os.listdir(path):
				filename = os.path.join(path,j)
				if os.path.isfile(filename):
					sortFile(filename)


if __name__ == '__main__':
	#sortAll("QueryLevelNorm")
	if len(sys.argv) < 2:
		sys.exit(-1)
	
	sortAll(sys.argv[1])
