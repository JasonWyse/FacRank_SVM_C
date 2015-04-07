#coding=utf-8
import MySQLdb
 

class DB(object):
	__instance = None
	def __init__(self):
		#set default value for mysql
		self.conn = None
		self.isConnected = False
		self.config = {'host':'172.22.0.26',
						'user':'root',
						'passwd':'root',
						'db':'benchmark',
						'port':'3306'}
		self.loadConfig()
	def __new__(cls, *args, **kwargs): 
		if not cls.__instance:  
			cls.__instance = super(DB, cls).__new__(cls, *args, **kwargs)  
		return cls.__instance 
	
	@staticmethod
	def getInstance():
		if not DB.__instance:
			DB.__instance = DB()
		return DB.__instance

	def loadConfig(self):
		"""
		read config file
		"""
		with open('benchmark.cfg','r') as fconfig:
			flag = False
			for line in fconfig:
				line = line.strip('\r').strip('\n').strip()
				if line.startswith('#')  or len(line) == 0:
					continue
				elif line == '[mysql]':
					flag = True
					continue
				elif line.startswith('[') and line.endswith(']'):
					flag = False
				elif flag:
					tmp = line.strip('\n').strip('\r').split('=')
					if len(tmp) != 2:
						continue
					self.config[tmp[0]] = tmp[1]

	def connect(self):
		try:
			if not self.isConnected:
				self.conn = MySQLdb.connect(host=self.config['host'],
											user=self.config['user'],
											passwd=self.config['passwd'],
											db=self.config['db'],
											port=int(self.config['port']))
				self.isConnected = True
		except MySQLdb.Error,e:
			raise Exception("Mysql Error %d: %s" % (e.args[0], e.args[1]))


	def execute(self,sql,args=None):
		flag = 0
		result = None
		try:
			cur = self.conn.cursor()
			flag = cur.execute(sql,args)
			result = cur.fetchall()
			cur.close()
		except MySQLdb.Error,e:
			raise Exception("Mysql Error %d: %s" % (e.args[0], e.args[1]))
		finally:
			return flag,result

	def commit(self):
		try:
			self.conn.commit()
		except MySQLdb.Error,e:
			raise Exception("Mysql Error %d: %s" % (e.args[0], e.args[1]))

	def close(self):
		try:
			self.conn.close()
			self.isConnected = False
		except MySQLdb.Error,e:
			raise Exception("Mysql Error %d: %s" % (e.args[0], e.args[1]))



class Entity(object):

	def __init__(self):
		self.db = DB.getInstance()
		self.values = {}

	def getIdByName(self,table,name):
		sql = "select id from %s where name = '%s'" %(table,name)
		flag,result = self.db.execute(sql)
		if flag == 1:
			return result[0][0]
		else:
			raise Exception('Error: no such raw with name %s in table %s,or more than one'%(name,table))
	
	def persist(self):
		pass
	#values是一个map，表示要表里的各个字段的键值对，若没有则会采用默认值，但是所有外键必须有（可以输入外键所在表的name值）

	def setValues(self,values):
		for (k,v) in  values.items():
			if not self.setValue(k,v):
				return False
		return True
	def setValue(self,key,value):
		if key == 'id' or key == 'creation_time':
			return False
		self.values[key] = value 
		return True



class L2r(Entity):
	""" the entity of L2r table  """
	def __init__(self):
		Entity.__init__(self)
		self.values={
			'id':'UUID()',
			'dataset_id':'','fold_name':'',
			'subset_name':'','baseline_id':'',
			'software_id':'','p_1':'NULL',
			'p_2':'NULL','p_3':'NULL',
			'p_4':'NULL','p_5':'NULL',
			'p_6':'NULL','p_7':'NULL',
			'p_8':'NULL','p_9':'NULL',
			'p_10':'NULL','map':'NULL',
			'ndcg_1':'NULL','ndcg_2':'NULL',
			'ndcg_3':'NULL','ndcg_4':'NULL',
			'ndcg_5':'NULL','ndcg_6':'NULL',
			'ndcg_7':'NULL','ndcg_8':'NULL',
			'ndcg_9':'NULL','ndcg_10':'NULL',
			'mean_ndcg':'NULL','run_time':'NULL',
			'hyper_paras':'NULL','env_desc':'NULL',
			'creation_time':'curdate()'
		}


			
	def setValue(self,key,value):
		if key == 'id' or key == 'creation_time':
			return False
		self.db.connect()
		if key == 'dataset_name':
			self.values['dataset_id'] = self.getIdByName('t_benchmark_dataset',value)
		elif key == 'baseline_name':
			self.values['baseline_id'] = self.getIdByName('t_benchmark_baseline',value)
		elif key == 'software_name':
			self.values['software_id'] = self.getIdByName('t_benchmark_software',value)
		else:
			if not self.values.has_key(key):
				self.db.close()
				return False
			else:
				self.values[key] = value;
		self.db.close()
		return True
			
	def persist(self):
		keys = []
		values = []
		for (k,v) in  self.values.items():
			if k != 'id' and (k.endswith('id') or k.endswith('name') or k.endswith('desc') or k.endswith('paras')):
				if not( v.startswith('\'') and v.endswith('\'')):
					v = "\'" + v + "\'"
			keys.append(k)
			values.append(v)
		if len(keys) != 32 or len(values) != 32:
			raise Exception('database table insert error')
		tmp = tuple(keys+values)
		preparedSql = "INSERT INTO t_benchmark_result_learning2rank(" +\
				"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" +\
				"VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
		sql = preparedSql %tmp
		self.db.connect()
		flag,result = self.db.execute(sql)
		self.db.commit()
		self.db.close()
		return flag
		#print sql
		#benchmark
	#ids 是一个map，保存了能获取唯一记录的键值对，包括主键和各个外键，或者外键表的name
	def fetchOne(self,ids):
		self.db.connect()
		conditions = []
		for (key,value) in ids.items():
			if key == 'dataset_name':
				conditions.append("dataset_id='%s'"%self.getIdByName('t_benchmark_dataset',value))
			elif key == 'baseline_name':
				conditions.append("baseline_id='%s'"%self.getIdByName('t_benchmark_baseline',value))
			elif key == 'software_name':
				conditions.append("software_id='%s'"%self.getIdByName('t_benchmark_software',value))
			elif not self.values.has_key(key):
				raise Exception('database fetch error,no such column')
			else:
				#if key.endswith('id'):
				conditions.append("%s='%s'"%(key,value))

		keys = []
		sql = "SELECT ''"
		for key in self.values:
			keys.append(key)
			sql += "," + key + " "
		sql += "FROM t_benchmark_result_learning2rank where 1=1 "
		for condition in conditions:
			sql += "and " + condition + " " 

		flag,result = self.db.execute(sql)
		if flag > 1:
			self.db.close()
			raise Exception('database fetch error,more than one result')
		if flag != 1:
			return False

		for i in range(0,len(keys)):
			self.values[keys[i]] = result[i+1]
		#print self.values

		self.db.close()
		return True
	

	#修改数据库，修改前将数据存入values中
	def merge(self):
		updates = []
		self.values['creation_time'] = 'curdate()'
		sql = "UPDATE t_benchmark_result_learning2rank SET "
		for (k,v) in  self.values.items():
			if k != 'id':
				if k.endswith('id') or k.endswith('name')  or k == 'hyper_paras' or k == 'env_desc':
					if not( v.startswith('\'') and v.endswith('\'')):
						v = "\'" + str(v) + "\'"
				elif v is None:
					v = 'NULL'
				sql += k + "=" + str(v) + " , "
		sql = sql[:-2]
		sql += " where id = " + "\'" + self.values['id'] +"\'"
		#print sql

		self.db.connect()
		flag,result = self.db.execute(sql)
		self.db.commit()
		self.db.close()
		return flag




class Recomm(Entity):
	""" the entity of Recomm table  """
	def __init__(self):
		Entity.__init__(self)
		self.values={
			'id':'UUID()',
			'dataset_id':'','fold_name':'',
			'subset_name':'','baseline_id':'',
			'software_id':'','p_1':'NULL',
			'p_2':'NULL','p_3':'NULL',
			'p_4':'NULL','p_5':'NULL',
			'rmse':'NULL','map':'NULL',
			'ndcg_1':'NULL','ndcg_2':'NULL',
			'ndcg_3':'NULL','ndcg_4':'NULL',
			'ndcg_5':'NULL','mean_ndcg':'NULL',
			'run_time':'NULL','hyper_paras':'NULL',
			'env_desc':'NULL','creation_time':'curdate()'
		}

	#values是一个map，表示要表里的各个字段的键值对，若没有则会采用默认值，但是所有外键必须有（可以输入外键所在表的name值）

	def setValue(self,key,value):
		if key == 'id' or key == 'creation_time':
			return False
		self.db.connect()
		if key == 'dataset_name':
			self.values['dataset_id'] = self.getIdByName('t_benchmark_dataset',value)
		elif key == 'baseline_name':
			self.values['baseline_id'] = self.getIdByName('t_benchmark_baseline',value)
		elif key == 'software_name':
			self.values['software_id'] = self.getIdByName('t_benchmark_software',value)
		else:
			if not self.values.has_key(key):
				self.db.close()
				return False
			else:
				self.values[key] = value;
		self.db.close()
		return True
			
	def persist(self):
		keys = []
		values = []
		for (k,v) in  self.values.items():
			if k != 'id' and (k.endswith('id') or k.endswith('name') or k.endswith('desc') or k.endswith('paras')):
				if not( v.startswith('\'') and v.endswith('\'')):
					v = "\'" + v + "\'"
			keys.append(k)
			values.append(v)
		if len(keys) != 23 or len(values) != 23:
			raise Exception('database table insert error')
		tmp = tuple(keys+values)
		preparedSql = "INSERT INTO t_benchmark_result_recomm(" +\
				"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" +\
				"VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
		sql = preparedSql %tmp
		self.db.connect()
		flag,result = self.db.execute(sql)
		self.db.commit()
		self.db.close()
		return flag

	#ids 是一个map，保存了能获取唯一记录的键值对，包括主键和各个外键，或者外键表的name
	def fetchOne(self,ids):
		self.db.connect()
		conditions = []
		for (key,value) in ids.items():
			if key == 'dataset_name':
				conditions.append("dataset_id='%s'"%self.getIdByName('t_benchmark_dataset',value))
			elif key == 'baseline_name':
				conditions.append("baseline_id='%s'"%self.getIdByName('t_benchmark_baseline',value))
			elif key == 'software_name':
				conditions.append("software_id='%s'"%self.getIdByName('t_benchmark_software',value))
			elif not self.values.has_key(key):
				raise Exception('database fetch error,no such column')
			else:
				#if key.endswith('id'):
				conditions.append("%s='%s'"%(key,value))

		keys = []
		sql = "SELECT ''"
		for key in self.values:
			keys.append(key)
			sql += "," + key + " "
		sql += "FROM t_benchmark_result_recomm where 1=1 "
		for condition in conditions:
			sql += "and " + condition + " " 
		#print sql
		flag,result = self.db.execute(sql)
		#print flag
		#print result
		if flag > 1:
			self.db.close()
			raise Exception('database fetch error,more than one result')
		if flag != 1:
			return False

		for i in range(0,len(keys)):
			self.values[keys[i]] = result[0][i+1]
		#print self.values

		self.db.close()
		return True


	def merge(self):
		updates = []
		self.values['creation_time'] = 'curdate()'
		sql = "UPDATE t_benchmark_result_recomm SET "
		for (k,v) in  self.values.items():
			if k != 'id':
				if k.endswith('id') or k.endswith('name') or k == 'hyper_paras' or k == 'env_desc':
					if not( v.startswith('\'') and v.endswith('\'')):
						v = "\'" + str(v) + "\'"
				elif v is None:
					v = 'NULL'
				sql += k + "=" + str(v) + " , "
		sql = sql[:-2]
		sql += " where id = " + "\'" + self.values['id'] +"\'"
		print sql
		self.db.connect()
		flag,result = self.db.execute(sql)
		self.db.commit()
		self.db.close()
		return flag





if __name__ == "__main__":
	"""
		example
	"""
	try:

		a = Recomm()
		values={
				'dataset_name':'movielens','fold_name':'test',
				'subset_name':'TRAIN','baseline_name':'MFwithbias',
				'software_name':'libmf','p_1':'NULL',
				'p_2':'NULL','p_3':'NULL',
				'p_4':'NULL','p_5':'NULL',
				'rmse':'0.78','run_time':'NULL',
				'hyper_paras':'bbbbbbbb','env_desc':'\'cccccccccccccc\'',
				}
		a.fetchOne({'fold_name':'test'})
		a.setValues(values)
		print a.merge()    
		#if a.fetchOne({'subset_name':'train',\
		#		'software_name':'libmf','dataset_name':'MOVIELENS',\
		#		'fold_name':'FOLD1','baseline_name':'MFrwithbias'}):
		#	print 'fetchOK'
		#print a.values
		#a.setValue('fold_name','FOLD1')
		#a.merge()
	except Exception , e:
		print e







#a.fetchOneByIds({'id':'1',"aaaaid":'dasd'})
#print dbop.getIdByName('t_domain',r"aaaa")



#print len(((1,2,4,5),(5,6,7),(1,2,4)))