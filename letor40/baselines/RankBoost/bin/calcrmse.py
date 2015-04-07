import mysql
import rmse
import os


values={}
entity = mysql.Recomm()
source = '../../../dataset/standard_1'
output = '../data/standard_1'

def init():
	values={
		'benchmark_name':'recommendation1.0',
		'dataset_name':'movielens1.0','fold_name':'FOLD1',
		'subset_name':'TRAIN','baseline_name':'MFwithbias',
		'software_name':'libmf',}

	entity.setValues(values)

def run():
	for i in range(1,6):
		fold = 'fold' + str(i)
		command = "cut " + source + "/" + fold + "/test" + " -d' ' -f 3 | paste -d ' ' - " + output + "/" + fold + "/test.predict  > tmp.txt"
		os.popen(command)
		r = rmse.rmse(r'tmp.txt')
		entity.setValue('fold_name','FOLD'+str(i))
		entity.setValue('subset_name','TEST')
		entity.setValue('rmse',str(r))
		print entity
		entity.persist()

































if __name__=='__main__':
	init()
	run()	
