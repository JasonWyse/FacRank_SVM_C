import os
import commands
import subprocess
import sys


iMaxPosition=10
if not "G:\\benchmark\\learning_to_rank\\letor40\\evaluation\\" in sys.path:
    sys.path.append("G:\\benchmark\\learning_to_rank\\letor40\\evaluation\\")
if not 'evaluate' in sys.modules:
    evaluate = __import__('evaluate')
else:
    eval('import evaluate')
    evaluate= eval('reload(evaluate)')


dataNum=raw_input('pls enter dataNum(2007 or2008)\n')
Measure=raw_input('pls enter measure (MAP/NDCG):\n')
path_adaRank='G:\\benchmark\\learning_to_rank\\letor40\\baselines\\AdaRank\\'
path_data='G:\\benchmark\\learning_to_rank\\letor40\\datasets\\mq%s\\'%dataNum
path_eval='G:\\benchmark\\learning_to_rank\\letor40\\evaluation\\evaluation.prl'

path_train=os.path.join(path_adaRank,'source_code\\AdaRank_CSharp.exe')
model=os.path.join(path_adaRank,'data\\model')
score=os.path.join(path_adaRank,'data\\score')
bestEval=os.path.join(path_adaRank,'eval\\mq%s_'%dataNum+Measure)
output=os.path.join(path_adaRank,'data\\output')


#bestEval=os.path.join(path_adaRank,'eval\\mq%s_'%dataNum+Measure+'trainround')


command_train=path_train+' train -data %s -model '+model+' -round %d -measure '+str(Measure)
command_test=path_train+' test -data %s -model '+model+' -output '+score+' -round %d'
#command_eval='perl '+path_eval+' %s '+score+' %s %s %d %d'

#ROUNDS=[0]*5
Map={}
PrecAtN={}
MeanNdcg={}
Ndcg={}

for fold_number in [1,2,3,4,5]:
#for fold_number in [1]:
	train=os.path.join(path_data,'Fold%d\\train.txt'%fold_number)
	
	test=os.path.join(path_data,'Fold%d\\test.txt'%fold_number)
	vali=os.path.join(path_data,'Fold%d\\vali.txt'%fold_number)
	
	os.system(command_train%(train,100))
	
	MAP=0
	for round in range(94,95,1):
		os.system(command_test%(vali,round))
		#command_test%(vali,round)
		#map=subprocess.check_output(command_eval%(vali,output,0,fold_number,0),shell=True)
		hsQueryDocLabelScore=evaluate.ReadInputFiles(vali, score)
		hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
		#print hsQueryEval
		map=evaluate.FoldMap(hsQueryEval)
		#map=subprocess.check_output(command_eval%(vali,output,0,round,0),shell=True)
		#print command_eval%(vali,output,0,round,0)
		print round,map
		if map>MAP:
			MAP=map
			BestRound=round
	
	ipos1='Fold%d'%fold_number	
	ipos2=1
	for dataType in [test,vali,train]:
		os.system(command_test%(dataType,BestRound))
		hsQueryDocLabelScore=evaluate.ReadInputFiles(dataType, score)
		hsQueryEval = evaluate.EvalQuery(hsQueryDocLabelScore)
		PrecAtN[ipos1+str(ipos2)]=evaluate.FoldPrecAtN(hsQueryEval)
		Map[ipos1+str(ipos2)]=evaluate.FoldMap(hsQueryEval)
		Ndcg[ipos1+str(ipos2)]=evaluate.FoldNdcg(hsQueryEval)
		MeanNdcg[ipos1+str(ipos2)]=evaluate.FoldMeanNdcg(hsQueryEval)
		ipos2+=1
	
	print 'rounds of best map are:%s'%BestRound
	

try:
	FOUT=open(bestEval,'w')
finally:
	if not FOUT:
		print 'Invalid command line.\n'
		print 'Open \$fnOut\' failed.\n'
		FOUT.close()
		exit -2
FOUT.write('Algorithm: AdaRank\nDataset:   MQ%s\n'%dataNum)

format={'1':'Performance on testing set\n','2':'Performance on validation set\n','3':'Performance on training set\n'}

for ipos2 in ['1','2','3']:
#for ipos2 in ['2']:
	FOUT.write(format[ipos2])
	FOUT.write('Folds\tNDCG@1\tNDCG@2\tNDCG@3\tNDCG@4\tNDCG@5\tNDCG@6\tNDCG@7\tNDCG@8\tNDCG@9\tNDCG@10\tMeanNDCG\n')
	#for fold_number in [1,2,3,4,5]:
	for fold_number in [1]:
		FOUT.write('Fold%d\t'%fold_number)
		ipos1='Fold%d'%fold_number
		NDCG=Ndcg[ipos1+ipos2]
		for i in range(iMaxPosition):
			FOUT.write('%.4f\t'%NDCG[i])
		FOUT.write('%.4f\t\n\n'%(MeanNdcg[ipos1+ipos2]))
	FOUT.write('Folds\tP@1\tP@2\tP@3\tP@4\tP@5\tP@6\tP@7\tP@8\tP@9\tP@10\tMAP\n')
	for fold_number in [1,2,3,4,5]:
	#for fold_number in [1]:
		FOUT.write('Fold%d\t'%fold_number)
		ipos1='Fold%d'%fold_number
		for i in range(iMaxPosition):
			FOUT.write('%.4f\t'%PrecAtN[ipos1+ipos2][i])
		FOUT.write('%.4f\t\n\n'%Map[ipos1+ipos2])