#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
void print_null(const char *s) {}

void exit_with_help()
{
	printf(
#ifdef FIGURE56
	"Usage: train [options] training_set_file testing_set_file\n"
#else
	"Usage: train [options] training_set_file [model_file]\n"
#endif
	"options:\n"
	"-s type : set type of solver (default 0)\n"
	"	0 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	1 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	2 -- L2-regularized L1-loss support vector regression (dual)\n"
	"	3 -- L2-regularized L2-loss rankSVM (direct count)\n"
	"	4 -- L2-regularized L2-loss rankSVM (y red-black tree)\n"
	"	5 -- L2-regularized L2-loss rankSVM (wx red-black tree)\n"
	"	6 -- L2-regularized L2-loss rankSVM (selection tree)\n"
	"	7 -- L2-regularized L2-loss rankSVM (y AVL tree)\n"
	"	8 -- L2-regularized L2-loss rankSVM (y AA tree)\n"
	"	9 -- PRSVM+\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 1, 3, 4, 5, 6, 7, 8, and 9\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 2\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}
//---------------------------- global variables -------------------------------
static char *line = NULL;
static int max_line_len;
//---------------------------- global variables -------------------------------
static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name);
void _parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name);
void read_problem(const char *filename);
#ifdef FIGURE56
void read_problem_test(const char *filename);
#endif
//---------------------------- global variables -------------------------------
struct feature_node *x_space;//x_space:global variable where stores all instances' features,
struct parameter param;//param: stores all input parameters got from commond line
struct _parameter _param;
struct problem prob;//prob: is a struct which records all information(all instances' label, features, queries...) about the input file
struct model* model_;//model_: stores the model we have trained
//---------------------------- global variables -------------------------------
int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	
#ifdef FIGURE56
	char test_file_name[1024];
	parse_command_line(argc, argv, input_file_name, test_file_name);
#else
	parse_command_line(argc, argv, input_file_name, model_file_name);//initialize global struct param, according to commond line 
	//_parse_command_line(argc, argv, input_file_name, model_file_name);//initialize global struct param, according to commond line 
#endif
	read_problem(input_file_name);//get all possible information about the train file into global struct prob
#ifdef FIGURE56
	read_problem_test(test_file_name);
#endif
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	//	struct model
//{
//	struct parameter param;
//	int nr_class;		/* number of classes */
//	int nr_feature;
//	double *w;
//	int *label;		/* label of each class */
//};
//	model_=train(&prob, &param);
//--------apply memory for V matrix--------------
	int i=0;
	double * p = Malloc(double,param.col_size * prob.l);
	srand( (unsigned)time( NULL ) );  //种子函数
	for (i=0;i<param.col_size * prob.l;i++)
	{		
		p[i]=rand()/(RAND_MAX+1.0);  //产生随机数的函数
	}
	double ** v_pp = Malloc(double* ,prob.l);
	param.v_pp = v_pp;
	
	for (i=0;i<prob.l;i++)
		param.v_pp[i] = &p[param.col_size * i];
	model_=_train(&prob, &param);

#ifdef FIGURE56
#else
	if(save_model(model_file_name, model_))
	{
		fprintf(stderr,"can't save model to file %s\n",model_file_name);
		exit(1);
	}
#endif
	free_and_destroy_model(&model_);
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(prob.query);
	free(x_space);
	////////free the variable
	free(v_pp);
	free(p);
#ifdef FIGURE56
	free(probtest.y);
	free(probtest.x);
	free(x_spacetest);
#endif
	free(line);
	return 0;
}

#ifdef FIGURE56
void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name)
#else
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
#endif
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = AVLTREE ;//L2R_L2LOSS_SVC
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	param.col_size = 50;//default
	param.ite = 200;//default
	param.eta = 1e-6;//default
	param.batch_size = 10;
	
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'C':
				param.C = atof(argv[i]);
				break;

			case 'c':
				param.col_size = (int)atof(argv[i]);
				break;

			case 'l':
				param.col_size =(int) atof(argv[i]);
				break;
			
			case 'b':
				param.batch_size =(int) atof(argv[i]);
				break;
			
			case 'i':
				param.ite = (int)atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eta = atof(argv[i]);
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();
		
	strcpy(input_file_name, argv[i]);
#ifdef FIGURE56
	if(i<argc-1)
		strcpy(test_file_name,argv[i+1]);
	else
	{
		exit_with_help();
	}
#else
	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		//strrchr() 函数查找字符在指定字符串中从后面开始的第一次出现的位置，如果成功，则返回从该位置到字符串结尾的所有字符，如果失败，则返回 false。
		//与之相对应的是strchr()函数，它查找字符串中首次出现指定字符的位置。
		char *p = strrchr(argv[i],'/');
		if(p==NULL)//there are no parent directories in the path of train data, that means train data are in the current directory
			p = argv[i];
		else//the train data contain parent directoryis, ++p: the pointer move from char '/' to the first char of the input train data
			++p;
		sprintf(model_file_name,"%s.model",p);//int sprintf ( char * str, const char * format, ... );Write formatted data to string
	}
#endif

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
			case WX_RBTREE:
			case Y_RBTREE:
			case AVLTREE:
			case AATREE:
			case DIRECT_COUNT:
			case SELECTION_TREE:
			case PRSVMP:
				param.eps = 0.001;
				break;
			case L2R_L1LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

// read in a problem (in libsvm format)

//struct feature_node
//{
//	int index;
//	double value;
//};
//void _parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
//#endif
//{
//	int i;
//	void (*print_func)(const char*) = NULL;	// default printing to stdout
//
//	// default values
//	_param.solver_type = L2R_L2LOSS_SVC;
//	_param.C = 1;
//	_param.eps = INF; // see setting below
//	_param.p = 0;
//	_param.nr_weight = 0;
//	_param.weight_label = NULL;
//	_param.weight = NULL;
//	_param.ite = 100; //default
//	// parse options
//	for(i=1;i<argc;i++)
//	{
//		if(argv[i][0] != '-') break;
//		if(++i>=argc)
//			exit_with_help();
//		switch(argv[i-1][1])
//		{
//			case 's':
//				_param.solver_type = atoi(argv[i]);
//				break;
//
//			case 'c':
//				_param.C = atof(argv[i]);
//				break;
//
//			case 'p':
//				_param.p = atof(argv[i]);
//				break;
//
//			case 'e':
//				_param.eps = atof(argv[i]);
//				break;
//
//			case 'i':
//				_param.ite = atof(argv[i]);
//				break;
//
//			case 'q':
//				print_func = &print_null;
//				i--;
//				break;
//
//			default:
//				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
//				exit_with_help();
//				break;
//		}
//	}
//
//	set_print_string_function(print_func);
//
//	// determine filenames
//	if(i>=argc)
//		exit_with_help();
//
//	strcpy(input_file_name, argv[i]);
//#ifdef FIGURE56
//	if(i<argc-1)
//		strcpy(test_file_name,argv[i+1]);
//	else
//	{
//		exit_with_help();
//	}
//#else
//	if(i<argc-1)
//		strcpy(model_file_name,argv[i+1]);
//	else
//	{
//		//strrchr() 函数查找字符在指定字符串中从后面开始的第一次出现的位置，如果成功，则返回从该位置到字符串结尾的所有字符，如果失败，则返回 false。
//		//与之相对应的是strchr()函数，它查找字符串中首次出现指定字符的位置。
//		char *p = strrchr(argv[i],'/');
//		if(p==NULL)//there are no parent directories in the path of train data, that means train data are in the current directory
//			p = argv[i];
//		else//the train data contain parent directoryis, ++p: the pointer move from char '/' to the first char of the input train data
//			++p;
//		sprintf(model_file_name,"%s.model",p);//int sprintf ( char * str, const char * format, ... );Write formatted data to string
//	}
//#endif
//
//	if(_param.eps == INF)
//	{
//		switch(_param.solver_type)
//		{
//			case L2R_L2LOSS_SVC:
//				_param.eps = 0.01;
//				break;
//			case L2R_L2LOSS_SVR:
//			case WX_RBTREE:
//			case Y_RBTREE:
//			case AVLTREE:
//			case AATREE:
//			case DIRECT_COUNT:
//			case SELECTION_TREE:
//			case PRSVMP:
//				_param.eps = 0.001;
//				break;
//			case L2R_L1LOSS_SVR_DUAL:
//				_param.eps = 0.1;
//				break;
//		}
//	}
//}
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);
	// struct problem
	// {
	// int l, n;//l: total instance number(starts from 0);n:total feature number
	// int *query;
	// double *y;//y:label value
	// struct feature_node **x;
	// };
	//prob.l: instances number of input file
	//prob.y: label array of each instance of input file
	//prob.x: pointer array of each instance's features
	//prob.query:query array of each instance of input file
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	prob.query = Malloc(int,prob.l);
	//x_space:apply enough spaces to store all instances' features
	x_space = Malloc(struct feature_node,elements+prob.l);
	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)//iterate all instances
	{
		prob.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			if (!strcmp(idx,"qid"))//qid
			{
				errno = 0;
				prob.query[i] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
			else//feature
			{
				errno = 0;
				x_space[j].index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_space[j].index;

				errno = 0;
				x_space[j].value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++j;
				/*for debug
				if(j==46)
					printf("%d",j);*/
			}
		}//finish parsing one line of data

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		x_space[j++].index = -1;
	}// iterate all instances
	prob.n=max_index;
	fclose(fp);
}

#ifdef FIGURE56
void read_problem_test(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	probtest.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		probtest.l++;
	}
	rewind(fp);

	probtest.y = Malloc(double,probtest.l);
	probtest.x = Malloc(struct feature_node *,probtest.l);
	probtest.query = Malloc(int,probtest.l);
	x_spacetest = Malloc(struct feature_node,elements+probtest.l);
	max_index = 0;
	j=0;
	for(i=0;i<probtest.l;i++)
	{
		probtest.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		probtest.x[i] = &x_spacetest[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		probtest.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			if (!strcmp(idx,"qid"))
			{
				errno = 0;
				probtest.query[i] = (int) strtol(val, &endptr,10);
				if(endptr == val || errno !=0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);
			}
			else
			{
				errno = 0;
				x_spacetest[j].index = (int) strtol(idx,&endptr,10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x_spacetest[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_spacetest[j].index;

				errno = 0;
				x_spacetest[j].value = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++j;
			}
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		x_spacetest[j++].index = -1;
	}
	probtest.n=max_index;
	fclose(fp);
}
#endif
