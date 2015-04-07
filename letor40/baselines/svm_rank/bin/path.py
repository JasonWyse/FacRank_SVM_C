#coding=utf-8
import os

class PathManager(object):
	"""
		This class manage the path of the project,the struct follows:
					                                                  $6
																 |---bin
																 |    $7
																 |---source_code   $8
													$5			 |		    |----dataset1
									  |-----baseline_software1---|---data---|
							$1		  |							 |    		|----dataset2   $9
					|-----baselines---|-----baseline_software2	 |                   |---dataset1
					|				  |							 |---localdatasets---|
					|				  |-----baseline_software3   |               $10 |---dataset2
					| 											 |			|---dataset1		
					|											 |---eval---|			
		home_path---| 			 	 |---orginal				 |			|---dataset2
					| 		$2		 |		$4					 |              $11
					|-----datasets---|---dataset1				 |		   |---dataset1
					|				 |							 |---log---|
					| 				 |---dataset2				           |---dataset2 	
					|       $3
					|-----evaluation
	"""

	def __init__(self,dataset,baseline,software=None,home_path=None,depth=3):
		if home_path is None:
			self.home_path = os.getcwd().rsplit(os.sep,depth)[0]	
		else:
			self.home_path = home_path
		self.dataset = dataset
		self.baseline = baseline
		self.software = software
		if software is None:
			self.algorithm  = baseline
		else:
			self.algorithm = "%s_%s"%(baseline,software)
		self.__path()


	def __path(self):
		#$1
		self.path_baselines = os.path.join(self.home_path,'baselines')  
		#$3   
		self.path_evaluation = os.path.join(self.home_path,'evaluation')
		#$2
		self.path_datasets = os.path.join(self.home_path,'datasets')
		#$4
		self.path_dataset = os.path.join(self.path_datasets,self.dataset)       #current dataset
		#$5
		self.path_algorithm = os.path.join(self.path_baselines,self.algorithm) #current baseline homepath
		#$7
		self.path_code = os.path.join(self.path_algorithm,'source_code') 
		#$8       
		self.path_data = os.path.join(self.path_algorithm,'data',self.dataset)
		#$9
		self.path_localdata = os.path.join(self.path_algorithm,'localdatasets',self.dataset)
		#$10
		self.path_eval = os.path.join(self.path_algorithm,'eval',self.dataset)
		#$11
		self.path_log = os.path.join(self.path_algorithm,'log',self.dataset)
		#$6
		self.path_bin = os.path.join(self.path_algorithm,'bin')


		self.__ifNotExistMk(self.path_data)
		self.__ifNotExistMk(self.path_localdata)
		self.__ifNotExistMk(self.path_eval)
		self.__ifNotExistMk(self.path_log)


	def __ifNotExistMk(self,path):
		if not os.path.exists(path):
			os.makedirs(path)


	def getPath(self,name):
		if not name.startswith('path'):
			return None
		result = eval('self.%s'%name)
		return result


if __name__ == '__main__':
	a = PathManager('ml-10m','mfwithbias','libmf-1.0.0')
	print a.getPath('path_log')
