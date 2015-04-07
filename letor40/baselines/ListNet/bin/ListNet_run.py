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
	rate=pars['rate']
	command_train='java -jar '+tls+' -train %s'+' -ranker %d'+' -save %s'+' -epoch %d'+' -lr %f'
	print os.popen(command_train%(train_set,7,model,r,rate))

#using testing command to create the score file of a given subset and its model
def predict_score(subset,model,score):
	tls=os.path.join(p.getPath('path_code'),tools_name)
	command_test='java -jar '+tls+' -load %s'+' -rank %s'+' -score %s'
	print os.popen(command_test%(model,subset,score))

#evaluate the performances of vali set, based on the given measure function 
#return performance of the specified measure function
def vali_evaluate(vali,score,msr):
	#eval_msr={'MAP':'FoldMap','NDCG':'FoldNdcg'}
	#fold_msr=eval_msr.get(msr)
	hsQueryDocLabelScore=evaluate.ReadInputFiles(vali, score,dataset)
	hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
	if msr == 'MAP':
		msr_value=evaluate.FoldMap(hsQueryEval)
	elif msr == 'NDCG':
		msr_value=evaluate.FoldNdcg(hsQueryEval)
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
	predict_score(subset,model,score)
	perfms=vali_evaluate(subset,score,msr)
	return perfms
def measure(subset,best_model,fold_number,subset_name):
	best_score=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number,subset_name+'.score')
	predict_score(subset,best_model,best_score)
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
	#best_model=os.path.join(temp_pth_f,'model')
	#train_model(train_set,best_model,best_pars)
	return
	
def train(train_set,vali_set,fold_number,**others):
	pth_data=p.getPath('path_data')
	temp_pth_dataset=os.path.join(pth_data,'%s'%dataset)
	# 'temp_pth_dataset' keeps all the five folds ,in each fold, the corresponding best model or best parameters files stay
	temp_pth_dataset=pth_data
	if not os.path.exists(temp_pth_dataset):
		os.mkdir(temp_pth_dataset)
	temp_pth_fold=os.path.join(temp_pth_dataset,'Fold%d'%fold_number)
	if not os.path.exists(temp_pth_fold):
		os.mkdir(temp_pth_fold)
	final_best_model = os.path.join(temp_pth_fold,'final_model')
	ROUND=300
	RATE=0.00001
	best_perfms=0
	model=os.path.join(p.getPath('path_data'),'model')
	log('ROUND %d'%ROUND)
	for r in range(100,ROUND+1,20):
		for rate in [0.00001,0.0001,0.001]:
			#rate=RATE
			pars={'round':r,'rate':rate}

			train_model(train_set,model,pars)
			perfms=vali_measure(vali_set,model)
			if perfms> best_perfms:
				best_perfms=perfms
				best_pars={'round':r,'rate':rate}
				if os.path.isfile(model): 
					open(final_best_model,"wb").write(open(model,"rb").read()) 
	#only save the best parameters, the model has been saved in final_best_model
	save_best_model(train_set,fold_number,best_pars)
	return 

def test(test,vali,train,fold_number):
	best_model=os.path.join(p.getPath('path_data'),'Fold%d'%fold_number,'final_model')
	Dict={'test':test,'vali':vali,'train':train}
	for subset_name in ['test','vali','train']:
		
		measure(Dict[subset_name],best_model,fold_number,subset_name)
	
	return


	
def ListNet():
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
		test(test_set,vali_set,train_set,fold_number)
		log('Fold%d ends testing'%fold_number)

def OutputFiles():
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

def init_database(entity):
	values={
			'benchmark_id':'000b9770-842d-11e4-a536-bcaec51b9163',
			'dataset_name':'MQ2007','fold_name':'FOLD1',
			'subset_name':'train','baseline_id':'485f6a02-842c-11e4-a536-bcaec51b9163',
			'software_id':'f72d0ac6-842c-11e4-a536-bcaec51b9163',
			}

	entity.setValues(values)
	return

def OutputDatabase():
	log('begin to save results into database')
	
	entity = mysql.L2r()
	init_database(entity)
	for subset_name in ['test','vali','train']:
		for fold_number in [1,2,3,4,5]:
			ipos1='Fold%d'%fold_number
		
			NDCG=Ndcg[ipos1+subset_name]
			PREC=PrecAtN[ipos1+subset_name]
		
			values={
			'benchmark_id':'000b9770-842d-11e4-a536-bcaec51b9163',
			'dataset_name':'MQ2007','fold_name':ipos1,
			'subset_name':subset_name,'baseline_id':'485f6a02-842c-11e4-a536-bcaec51b9163',
			'software_id':'f72d0ac6-842c-11e4-a536-bcaec51b9163',	
			'p_1':'NULL','p_2':'NULL',
			'p_3':'NULL','p_4':'NULL',
			'p_5':'NULL','p_6':'NULL',
			'p_7':'NULL','p_8':'NULL',
			'p_9':'NULL','p_10':'NULL',
			'map':str(Map[ipos1+subset_name]),
			'ndcg_1':'NULL','ndcg_2':'NULL',
			'ndcg_3':'NULL','ndcg_4':'NULL',
			'ndcg_5':'NULL','ndcg_6':'NULL',
			'ndcg_7':'NULL','ndcg_8':'NULL',
			'ndcg_9':'NULL','ndcg_10':'NULL',
			'mean_ndcg':str(MeanNdcg[ipos1+subset_name])
				}
			for i in range(IMAXPOSITION):
				values['p_'+str(i+1)]=str(PREC[i])
				values['ndcg_'+str(i+1)]=str(NDCG[i])
			
			entity.setValues(values)	
			entity.persist()
	log('save results to database done')
	return
			
DT={'0':'Mq2007','1':'Mq2008','2':'OHSUMED'}
MSR={'0':'MAP','1':'NDCG'}
iMaxPosition=10
tools_name=os.path.join('RankLib-v2.1','bin','RankLib.jar')
algthm_name='ListNet'


dataset=DT.get(raw_input('pls choose dataset(0:Mq2007,1:Mq2008,2:OHSUMED):\n'))
msr=MSR.get(raw_input('pls choose measure (0:MAP,1:NDCG):\n'))

	
p=PathManager(dataset,algthm_name)



Map={}
PrecAtN={}
MeanNdcg={}
Ndcg={}


		
saveout=sys.stdout
__start()


ListNet()

OutputFiles()


__end()
sys.stdout=saveout



#OutputDatabase()





