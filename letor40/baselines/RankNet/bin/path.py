import os


class PathManager(object):

	def __init__(self,dataset,algorithm,home_path=None,baseline=None,agrv=3):
		if home_path is None:
			self.home_path = os.getcwd().rsplit(os.sep,agrv)[0]	
		else:
			self.home_path = home_path
		self.dataset = dataset
		self.algorithm  = algorithm
		self.baseline = baseline
		self.__path()


	def __path(self):
		self.path_baselines = os.path.join(self.home_path,'baselines')     
		self.path_evalut = os.path.join(self.home_path,'evaluation')
		self.path_datasets = os.path.join(self.home_path,'datasets')
		self.path_dataset = os.path.join(self.path_datasets,self.dataset)       #current dataset
		self.path_algorithm = os.path.join(self.path_baselines,self.algorithm) 
		self.path_code = os.path.join(self.path_algorithm,'source_code')        
		self.path_data = os.path.join(self.path_algorithm,'data')
		self.path_localdata = os.path.join(self.path_algorithm,'localdatasets')
		self.path_eval = os.path.join(self.path_algorithm,'eval')
		self.path_log = os.path.join(self.path_algorithm,'log')
		self.path_bin = os.path.join(self.path_algorithm,'bin')
		if not self.baseline is None:
			self.path_data = os.path.join(self.path_data,self.baseline)
			self.path_localdata = os.path.join(self.path_localdata,self.baseline)
			self.path_eval = os.path.join(self.path_eval,self.baseline)
			self.path_log = os.path.join(self.path_log,self.baseline)
			self.path_bin = os.path.join(self.path_bin,self.baseline)


	def getPath(self,name):
		result = eval('self.%s'%name)
		return result


if __name__ == '__main__':
	a = PathManager('11','222')
	print a.getPath('path_log')