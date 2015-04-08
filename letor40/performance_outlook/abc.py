import sys
import os 
import shutil 


def cp(src,dst):
	if os.path.isfile(src): 
		open(dst, "wb").write(open(src, "rb").read())


def getBaselines(rootPath):
	files = []
	rootPath = os.path.abspath(rootPath)
	for i in os.listdir(rootPath):
		files.append(os.path.join(rootPath,i))
	return files

def cpOneFold(path,dst):
	if not os.path.isdir(path):
		return 
	for i in os.listdir(path):
		cp(os.path.join(path,i),os.path.join(dst,i))

def run(rootPath,dstPath):
	paths = getBaselines(rootPath)
	for path in paths:
		evalPath = os.path.join(path,'eval')
		for folds in os.listdir(evalPath):
			cpOneFold(os.path.join(evalPath,folds),dstPath)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		sys.exit(-1)
	if os.path.exists('output'):
		os.system('rm -rf output')
	if not os.path.exists('input'):
		os.mkdir('input');
	run(sys.argv[1],sys.argv[2])
	os.system("python result.py input output")