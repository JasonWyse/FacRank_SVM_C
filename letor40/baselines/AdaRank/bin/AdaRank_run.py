import os
import commands
import subprocess
import sys
#import path
import logging
import mysql
from path import PathManager
import time
import evaluate


def log(msg):
	print "[%s]: %s"%(__getTime(),msg)
def __getLogFileName():
	t = time.strftime('%m%d_%H%M',time.localtime(time.time()))
 	return os.path.join(p.getPath('path_log'),'%s.log'%t)

def __start():
	#self.__saveout = sys.stdout
	fsock = open(__getLogFileName(), 'w')
	sys.stdout = fsock

def __end():
	sys.stdout.close()  
	#sys.stdout = .__savaout
	

def __getTime():
	return time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))
	
def __ifNotExistMk(path):
	if not os.path.exists(path):
		os.makedirs(path)


def train_model(train_set,model,pars):
	tls=os.path.join(p.getPath('path_code'),tools_name)
	r=pars['round']
	command_train=tls+' train -data %s -model %s -round %d -measure %s'
	print os.popen(command_train%(train_set,model,r,msr))
def find_pars_value(name,fold_number):
	best_pars=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number,'bestPars')
	f_bestPars=open(best_pars,'r')
	line=f_bestPars.readline()
	while line!='':
		if line.split()[0]==name:
			value=line.split()[1]
			break
		else:
			line=f_bestPars.readline()
	return value
#using testing command to create the score file of a given subset and its model
def predict_score(subset,model,score,pars=None):
	tls=os.path.join(p.getPath('path_code'),tools_name)
	if pars!=None:
		r=pars['round']
	command_test=tls+' test -data %s -model %s -output %s -round %s'
	print os.popen(command_test%(subset,model,score,r))

#evaluate the performances of vali set, based on the given measure function 
#return performance of the specified measure function
def vali_evaluate(vali,score,msr,dataset):
	#eval_msr={'MAP':'FoldMap','NDCG':'FoldNdcg'}
	#fold_msr=eval_msr.get(msr)
	hsQueryDocLabelScore=evaluate.ReadInputFiles(vali, score,dataset)
	hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
	if msr == 'MAP':
		msr_value = evaluate.FoldMap(hsQueryEval)
	elif msr == 'NDCG':
		msr_value = evaluate.FoldNdcg(hsQueryEval)
	return msr_value

#evaluate the performances of each subset, in terms of several measure functions
#save these performances to some dictionaries
def results(subset,score,p,q):	
	hsQueryDocLabelScore=evaluate.ReadInputFiles(subset, score,dataset)
	hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
	PrecAtN[p+q]=evaluate.FoldPrecAtN(hsQueryEval)
	Map[p+q]=evaluate.FoldMap(hsQueryEval)
	Ndcg[p+q]=evaluate.FoldNdcg(hsQueryEval)
	MeanNdcg[p+q]=evaluate.FoldMeanNdcg(hsQueryEval)
	return
	

def vali_measure(subset,model,pars=None):
	score=os.path.join(p.getPath('path_data'),'score')
	predict_score(subset,model,score,pars)
	perfms=vali_evaluate(subset,score,msr,dataset)
	return perfms
def measure(subset,best_model,fold_number,subset_name,pars=None):
	best_score=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number,subset_name+'.score')
	predict_score(subset,best_model,best_score,pars)
	results(subset,best_score,'Fold%d'%fold_number,subset_name)
	return 
		
#save bestPars to files and train best model based on bestPars 	
def save_best_model(train_set,fold_number,best_pars):
	#make a dir named by fold number,
	#in order to save best pars and model of each dataset
	temp_pth_f=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number)
	__ifNotExistMk(temp_pth_f)
	#create a file named bestPars to save bestPars 
	F_pars=open(os.path.join(temp_pth_f,'bestPars'),'w')
	F_pars.write('%s\t%s\n'%('parameters','value'))
	for k,v in best_pars.items():
		F_pars.write('%s\t%s\n'%(k,v))	
	F_pars.close()
	#train best_model based on bestPars
	best_model=os.path.join(temp_pth_f,'model')
	train_model(train_set,best_model,best_pars)
	return

# for the fold indicated by fold_number, we use the vali_set to select the best model 	
def train(train_set,vali_set,fold_number,**others):
	# 'temp_pth_fold' its parent directory is 'temp_pth_dataset' 
		
	ROUND=200
	best_perfms=0
	model=os.path.join(p.getPath('path_data'),'model')
	log('ROUND %d'%ROUND)	
	pars={'round':ROUND}
	# for AdaRank ,we just need to train once time to get the weaker classifiers, and we use validation set to choose the the best classifiers combination
	train_model(train_set,model,pars)
	for r in range(1,ROUND+1,40):#range(1,5,2) from 1 to 5 ,internal 2(do not include 5)
	
		pars={'round':r}
		perfms=vali_measure(vali_set,model,pars)
		if perfms> best_perfms:
			best_perfms=perfms
			best_pars={'round':r}		
		save_best_model(train_set,fold_number,best_pars)
	return 

def test(test,vali,train,fold_number,best_model):
	#best_model=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number,'model')
	r=find_pars_value('round',fold_number)
	pars={'round':r}
	Dict={'test':test,'vali':vali,'train':train}
	for subset_name in ['test','vali','train']:

		measure(Dict[subset_name],best_model,fold_number,subset_name,pars)
	
	return


	
def CrossValidation():
	path_dataset=p.getPath('path_dataset')
	log('training starts')
	for fold_number in [1,2,3,4,5]:
		train_set=os.path.join(path_dataset,'Fold%d'%fold_number,'train.txt')
		vali_set=os.path.join(path_dataset,'Fold%d'%fold_number,'vali.txt')
		test_set=os.path.join(path_dataset,'Fold%d'%fold_number,'test.txt')
		
		log('Fold%d starts training'%fold_number)
		train(train_set,vali_set,fold_number)
		log('Fold%d ends training'%fold_number)
		
		log('Fold%d starts testing'%fold_number)
		best_model=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number,'model')
		test(test_set,vali_set,train_set,fold_number,best_model)
		log('Fold%d ends testing'%fold_number)

def OutputFiles(msr,dataset,algthm_name):
	log('begin to write results into files')

	bstEvl=os.path.join(p.getPath('path_eval'),'%s_%s_%s'%(msr,dataset,algthm_name))
#output the baselines to 'bstEvl' file		
	try:
		FOUT=open(bstEvl,'w')
	finally:
		if not FOUT:
			print 'Invalid command line.\n'
			print 'Open \$fnOut\' failed.\n'
			FOUT.close()
			exit -2
			
	FOUT.write('Validation measure: msr_%s\nDataset: %s\nAlgorithm: %s\n'%(msr,dataset,algthm_name))

	format={'test':'\nperformance on testing set\n','vali':'\nPerformance on validation set\n','train':'\nPerformance on training set\n'}

	for subset_name in ['test','vali','train']:
		FOUT.write(format[subset_name])
		FOUT.write('Folds\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\tNDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10\tMeanNDCG\n')
		for fold_number in [1,2,3,4,5]:
			FOUT.write('Fold%d\t'%fold_number)
			ipos1='Fold%d'%fold_number
			NDCG=Ndcg[ipos1+subset_name]
			for i in range(iMaxPosition):
				FOUT.write('%.4f\t'%NDCG[i])
			FOUT.write('%.4f\t\n'%(MeanNdcg[ipos1+subset_name]))
			
		FOUT.write('\nFolds\tP@1\tP@2\tP@3\tP@4\tP@5\tP@6\tP@7\tP@8\tP@9\tP@10\tMAP\n')
		for fold_number in [1,2,3,4,5]:
			FOUT.write('Fold%d\t'%fold_number)
			ipos1='Fold%d'%fold_number
			PREC=PrecAtN[ipos1+subset_name]
			for i in range(iMaxPosition):
				FOUT.write('%.4f\t'%PREC[i])
			FOUT.write('%.4f\t\n'%Map[ipos1+subset_name])
	log('writing files done')
	return

	
DT={'0':'Mq2007','1':'Mq2008','2':'OHSUMED'}
MSR={'0':'MAP','1':'NDCG'}
iMaxPosition=10
tools_name=os.path.join('AdaRank_CSharp.exe')
algthm_name='AdaRank'


dataset=DT.get(raw_input('pls choose dataset(0:Mq2007,1:Mq2008,2:OHSUMED):\n'))
msr=MSR.get(raw_input('pls choose measure (0:MAP,1:NDCG):\n'))

	
p=PathManager(dataset,algthm_name)



Map={}
PrecAtN={}
MeanNdcg={}
Ndcg={}

		
saveout=sys.stdout
__start()

CrossValidation()

OutputFiles(msr,dataset,algthm_name)


__end()
sys.stdout=saveout



#OutputDatabase()





