import os
import commands
import subprocess
import sys
#import path
import logging
import mysql
from path import PathManager
import time


#tuning parameter's process
def tune_paras(vali,score,msr):
	eval_msr={'MAP':'FoldMap','NDCG':'FoldNdcg'}
	fold_msr=eval_msr.get(msr)
	hsQueryDocLabelScore=evaluate.ReadInputFiles(vali, score, dataset)
	hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
	msr_value=evaluate.FoldMap(hsQueryEval)
	return msr_value

def results(subset,score,p,q):	
	hsQueryDocLabelScore=evaluate.ReadInputFiles(subset, score,dataset)
	hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
	PrecAtN[p+q]=evaluate.FoldPrecAtN(hsQueryEval)
	Map[p+q]=evaluate.FoldMap(hsQueryEval)
	Ndcg[p+q]=evaluate.FoldNdcg(hsQueryEval)
	MeanNdcg[p+q]=evaluate.FoldMeanNdcg(hsQueryEval)
	return
	
def validation():
	pth_data=p.getPath('path_data')
	pth_dataset=p.getPath('path_dataset')
	score=os.path.join(pth_data,'score')
	model=os.path.join(pth_data,'model')


# tune the parameters
	for fold_number in [1,2,3,4,5]:
		train=os.path.join(pth_dataset,'Fold%d'%fold_number,'train.txt')
	
		test=os.path.join(pth_dataset,'Fold%d'%fold_number,'test.txt')
		vali=os.path.join(pth_dataset,'Fold%d'%fold_number,'vali.txt')
		
		temp_pth_d=os.path.join(pth_data,'%s'%dataset)
		if not os.path.exists(temp_pth_d):
			os.mkdir(temp_pth_d)
		temp_pth_f=os.path.join(temp_pth_d,'Fold%d'%fold_number)
		if not os.path.exists(temp_pth_f):
			os.mkdir(temp_pth_f)
		final_best_model = os.path.join(temp_pth_f,'final_model')
		#perfms is the value of map or ndcg; Perfms is the best performance, used to choose the best paras
		Perfms=0
		
			
		for r in range(20,200+1,10):
			for layer in [1,2]:
				for node in range(5,20+1,2):
					for lr in [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]:
			#for rate in [0.00001,0.0001,0.001,0.01]:
						print os.popen(command_train%(train,1,model,r,layer,node,lr))
						print os.popen(command_test%(model,vali,score))
						perfms=tune_paras(vali,score,msr)
						if perfms> Perfms:
							Perfms=perfms
							BestRound=r
							Layer=layer
							Node=node
							Lr=lr
							if os.path.isfile(model): 
								open(final_best_model,"wb").write(open(model,"rb").read()) 	
				#Rate=rate
		
		F_pars=open(os.path.join(temp_pth_f,'bestPars'),'w')
		F_pars.write('Fold%d\n%s\t%s\t%s\t%s\n'%(fold_number,BestRound,Layer,Node,Lr))
		
		#save the best performance 
		ipos1='Fold%d'%fold_number	
		
		#ipos=0
		#
		best_model = final_best_model
		Dict={'test':test,'vali':vali,'train':train}
		for subset_name in ['test','vali','train']:
			best_score=os.path.join(temp_pth_f,subset_name+'_score')
			
			#print os.popen(command_train%(train,1,best_model,BestRound,Layer,Node,Lr))
			print os.popen(command_test%(best_model,Dict[subset_name],best_score))			
			results(Dict.get(subset_name),best_score,ipos1,subset_name)
			#ipos+=1
		f_t = time.strftime('%m%d_%H%M',time.localtime(time.time()))
		print 'Fold_%s finish at %s\t'%(fold_number,f_t)
	F_pars.close()
	return

def OutputFiles():
	#log('begin to write results into files')
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
	#log('writing files done')
	return

def init(entity):
	values={
			'benchmark_id':'000b9770-842d-11e4-a536-bcaec51b9163',
			'dataset_name':'MQ2007','fold_name':'FOLD1',
			'subset_name':'train','baseline_id':'485f6a02-842c-11e4-a536-bcaec51b9163',
			'software_id':'f72d0ac6-842c-11e4-a536-bcaec51b9163',
			}

	entity.setValues(values)
	return

def OutputDatabase():
	entity = mysql.L2r()
	init(entity)
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
			for i in range(iMaxPosition):
				values['p_'+str(i+1)]=str(PREC[i])
				values['ndcg_'+str(i+1)]=str(NDCG[i])
			
			entity.setValues(values)	
			entity.persist()
	return
			
DT={'0':'Mq2007','1':'Mq2008','2':'OHSUMED'}
Msr={'0':'MAP','1':'NDCG'}
#ipos2={'0':'test','1':'vali','2':'train'}
iMaxPosition=10

dataset=DT.get(raw_input('pls choose dataset(0:Mq2007,1:Mq2008,2:OHSUMED):\n'))
msr=Msr.get(raw_input('pls choose measure (0:MAP,1:NDCG):\n'))

tools_name='RankLib-v2.1\\bin\\RankLib.jar'
algthm_name='RankNet'
	
p=PathManager(dataset,algthm_name)

pth_bslns=p.getPath('path_baselines')
pth_evlt=p.getPath('path_evalut')
pth_dataset=p.getPath('path_dataset')
pth_data=p.getPath('path_data')

tls=os.path.join(p.getPath('path_code'),tools_name)
pth_eval=p.getPath('path_eval')
pth_log=p.getPath('path_log')

#import evaluate.py
if not pth_evlt in sys.path:
    sys.path.append(pth_evlt)
if not 'evaluate' in sys.modules:
    evaluate = __import__('evaluate')
else:
    eval('import evaluate')
    evaluate= eval('reload(evaluate)')

Map={}
PrecAtN={}
MeanNdcg={}
Ndcg={}

'''command_train=tls+' train -data %s -model %s -round %d -measure '+str(msr)
command_test=tls+' test -data %s -model %s -output %s -round %d'''

command_train='java -jar '+tls+'-train %s -ranker %d -save %s -epoch %d -layer %d -node %d -lr %f'
command_test='java -jar '+tls+' -load %s -rank %s'+' -score %s'
#message to be logged 	


	

try:
	#if LOGFILENAME is None:
	t = time.strftime('%m%d_%H%M',time.localtime(time.time()))
	LOGFILENAME=os.path.join(pth_log,'%s.log'%t)
	F_log=open(LOGFILENAME,'w')
finally:
	if not F_log:
		print 'Invalid command line.\n'
		print 'Open \flog\' failed.\n'
		F_log.close()
		exit -2
saveout=sys.stdout
sys.stdout=F_log
	

print 'pars\n%s\t%s\t%s\t%s\n'%('epoch','layer','node','learning_rate')
validation()

t_OutputFile = time.strftime('%m%d_%H%M',time.localtime(time.time()))
print '\nat %s begin to write into files..\t'%(t_OutputFile)
OutputFiles()

sys.stdout=saveout
F_log.close()

#outputDatabase(entity)