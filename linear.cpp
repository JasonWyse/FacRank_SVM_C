#include <math.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include "linear.h"
#include "tron.h"
#include "binarytrees.h"
#include "selectiontree.h"
//using namespace std;
#ifdef FIGURE56
struct feature_node *x_spacetest;
struct problem probtest;
#endif
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
#ifdef FIGURE56
void evaluate_test(double* w)
{
	int i;
	double *true_labels = Malloc(double,probtest.l);
	double *dec_values = Malloc(double,probtest.l);
	if(&probtest != NULL)
	{
		for(i = 0; i < probtest.l; ++i)
		{
			feature_node *x = probtest.x[i];
			true_labels[i] = probtest.y[i];
			double predict_label = 0;
			for(; x->index != -1; ++x)
				predict_label += w[x->index-1]*x->value;
			dec_values[i] = predict_label;
		}
	}
	double result[3];
	eval_list(true_labels, dec_values, probtest.query, probtest.l, result);
	info("Pairwise Accuracy = %g%%\n",result[0]*100);
	info("MeanNDCG (LETOR) = %g\n",result[1]);
	info("NDCG (YAHOO) = %g\n",result[2]);
	free(true_labels);
	free(dec_values);
}
#endif

static int compare_id_and_value(const void *a, const void *b)
{
	struct id_and_value *ia = (struct id_and_value *)a;
	struct id_and_value *ib = (struct id_and_value *)b;
	if(ia->value > ib->value)
		return -1;
	if(ia->value < ib->value)
		return 1;
	return 0;
}

class y_rbtree_rank_fun: public function
{
	public:
		y_rbtree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count);
		~y_rbtree_rank_fun();

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *s, double *Hs);
		int get_nr_variable(void);

	protected:
		void Xv(double *v, double *Xv);
		void XTv(double *v, double *XTv);

		double C;
		double *z;
		int *l_plus;
		int *l_minus;		
		double *alpha_plus;
		double *alpha_minus;
		const problem *prob;
		int nr_subset;
		int *perm;
		int *start;
		int *count;
		id_and_value **pi;
};

y_rbtree_rank_fun::y_rbtree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count)
{
	int l=prob->l;
	this->prob = prob;
	this->nr_subset = nr_subset;
	this->perm = perm;
	this->start = start;
	this->count = count;
	this->C = C;
	l_plus = new int[l];
	l_minus = new int[l];
	alpha_plus = new double[l];
	alpha_minus = new double[l];
	z = new double[l];
	pi = new id_and_value* [nr_subset];
	for (int i=0;i<nr_subset;i++)
		pi[i] = new id_and_value[count[i]];
}

y_rbtree_rank_fun::~y_rbtree_rank_fun()
{
	delete[] l_plus;
	delete[] l_minus;
	delete[] alpha_plus;
	delete[] alpha_minus;
	delete[] z;
	for (int i=0;i<nr_subset;i++)
		delete[] pi[i];
	delete[] pi;
}

double y_rbtree_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	rbtree *T;
	Xv(w,z);
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);
		T=new rbtree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new rbtree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}

void y_rbtree_rank_fun::grad(double *w, double *g)
{
	int i;
	int l=prob->l;
	double *ATAXw;
	ATAXw = new double[l];
	int w_size=get_nr_variable();
	for (i=0;i<l;i++)
		ATAXw[i]=(double)l_plus[i]-(double)l_minus[i]+((double)l_plus[i]+(double)l_minus[i])*z[i]-alpha_plus[i]-alpha_minus[i];
	XTv(ATAXw, g);//(X^T)(A^TAXw), get the result in g
	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*C*g[i];
	delete[] ATAXw;
}

int y_rbtree_rank_fun::get_nr_variable(void)
{
	return prob->n;
}

void y_rbtree_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *y=prob->y;
	double *wa = new double[l];
	rbtree *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new rbtree(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new rbtree(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(y[pi[i][j].id]);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

void y_rbtree_rank_fun::Xv(double *v, double *Xv)//w z
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}
//for all instances, get the sum of the product of each instance X and v
void y_rbtree_rank_fun::XTv(double *v, double *XTv)
{
	int i;
	int l = prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class y_avltree_rank_fun: public y_rbtree_rank_fun
{
	public:
		y_avltree_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count): y_rbtree_rank_fun(prob, C, nr_subset, perm, start, count){};
		double fun(double *w);
		double _fun(double *w);
		void _grad(model*model_,const problem* prob,int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL);
		void Hv(double *s, double *Hs);
};

double y_avltree_rank_fun::fun(double *w)
{
	int i,j,k;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	avl *T;
	Xv(w,z);//the product of w and each instance's feature,the result is a vector of length of l(total instances in the input file) stored in variable z;
	for (i=0;i<nr_subset;i++)//i:iterate each query group(one query group means one distict)
	{
		for (j=0;j<count[i];j++)//j: iterate every instance in each query group
		{
			//id_and_value **pi;
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		//void qsort( void *base, size_t num, size_t width, int (__cdecl *compare )
		//第一个参数 base 是 需要排序的目标数组名（或者也可以理解成开始排序的地址，因为可以写&s[i]这样的表达式）
		//第二个参数 num 是 参与排序的目标数组元素个数
		//第三个参数 width 是单个元素的大小（或者目标数组中每一个元素长度），推荐使用sizeof(s[0]）这样的表达式
		//第四个参数 compare 就是让很多人觉得非常困惑的比较函数啦。
		//典型的compare的定义是int compare(const void *a,const void *b);返回值必须是int，两个参数的类型必须都是const void *，
		//那个a,b是随便写的，个人喜好。假设是对int排序的话，如果是升序，那么就是如果a比b大返回一个正值，小则负值，相等返回0，其他的依次类推，后面有例子来说明对不同的类型如何进行排序。
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);//descend order 
		T=new avl(count[i]);//initialize the tree nodes
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new avl(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;	
	for(i=0;i<l;i++)//corresponding to formular 18
		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	return(f);
}
double y_avltree_rank_fun::_fun(double *w)
{
	int i,j,k;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	avl *T;	
	Xv(w,z);//the product of w and each instance's feature,the result is a vector of length of l(total instances in the input file) stored in variable z;
	double loss_term = 0;
	int pw = 0;
	for (i=0;i<nr_subset;i++)//i:iterate each query group(one query group means one distict)
	{
		for (j=0;j<count[i];j++)//j: iterate every instance in each query group
		{
			//id_and_value **pi;
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		//void qsort( void *base, size_t num, size_t width, int (__cdecl *compare )
		//第一个参数 base 是 需要排序的目标数组名（或者也可以理解成开始排序的地址，因为可以写&s[i]这样的表达式）
		//第二个参数 num 是 参与排序的目标数组元素个数
		//第三个参数 width 是单个元素的大小（或者目标数组中每一个元素长度），推荐使用sizeof(s[0]）这样的表达式
		//第四个参数 compare 就是让很多人觉得非常困惑的比较函数啦。
		//典型的compare的定义是int compare(const void *a,const void *b);返回值必须是int，两个参数的类型必须都是const void *，
		//那个a,b是随便写的，个人喜好。假设是对int排序的话，如果是升序，那么就是如果a比b大返回一个正值，小则负值，相等返回0，其他的依次类推，后面有例子来说明对不同的类型如何进行排序。
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);//descend order 
		T=new avl(count[i]);//initialize the tree nodes
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new avl(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	//calculate pw
	
	for(i=0;i<l;i++)
		pw += l_plus[i];	
	//calculate loss term of objective function
	for(i=0;i<l;i++)
		loss_term += (l_plus[i] - l_minus[i])*z[i];
	f += C*(pw-loss_term);	
	return(f);
}
void y_avltree_rank_fun::_grad(model *model_,const problem *prob,int nr_subset, int *perm, int *start, int *count)
{
	int batch_size = model_->param.batch_size;//batch_size = model_->param.batch_size;
	double  eta = model_->param.eta;;
	double ** v_pp = model_->param.v_pp;	
	int col_size = model_->param.col_size;
	feature_node ** x = prob->x;
	int l = prob->l;//all instances number
	int n = prob->n;//all feature number
	double * y = prob->y;
	double * w = model_->w;
	std::set<int> s;
	double * Y = Malloc(double,prob->n);
	double * batch_v_grad = Malloc(double,batch_size*col_size);
	double ** v_grad_pp = Malloc(double*,batch_size);//to store the gradient of batch_size lines of matrix V
	int * update_indexes = Malloc(int,batch_size);// to store the doc 
	double* minusResult = Malloc(double,n);
	double C = model_->param.C;
	//initialize batch_v_grad
	for(int i=0; i<batch_size*col_size;i++)
	{
		batch_v_grad[i] = 0;
	}
	//initialize v_grad_pp
	for(int i=0; i<batch_size;i++)
	{
		v_grad_pp[i] = &batch_v_grad[col_size*i];
	}
	//initialize Y
	for(int i=0; i<n; i++)
	{
		Y[i] = 0;
	}
    //calculate Y
	for(int i=0;i<l;i++)
	{
		for(int j=0;j<n;j++)
		{
			Y[j] += (l_plus[i]-l_minus[i])*x[i][j].value;
		}
	}
	// 
	while(1)//choose lines of batch_size of matrix V to update in gradient function
	{
		int r = rand() % prob->l;
		s.insert(r);
		if(s.size() == batch_size)
		{
			break;
		}
	}
    std::set<int>::iterator it; //定义前向迭代器 
    //中序遍历集合中的所有元素  
	int batch_count = 0;
    for(it=s.begin();it!=s.end();it++) 
	{
		int v_index = *it;
		int j;
		for(j=0;j<nr_subset-1;j++)//after the for loop, j is the query index which v_index belongs to 
		{
			if(start[j]<= v_index && start[j+1]> v_index)
				break;
		}
		double *phi_v_index_minus,*phi_v_index_plus,effi1,effi2;
		for(int k=0;k<count[j];k++)//iterate all docs of the query where v_index belongs to, to find preference pairs which v_index involves
		{
			int q_docID = start[j]+k;
			if (y[v_index]>y[q_docID])
			{
				feature_node_Minus(n, x[v_index], x[q_docID], minusResult);
				phi_v_index_minus = minusResult;
				effi1 = dotProduct(n, w, phi_v_index_minus);
				effi2 = dotProduct(n, phi_v_index_minus, Y);
				int p;
				for( p=0;p<col_size;p++)
					v_grad_pp[batch_count][p] += (effi1 - C*effi2) * v_pp[q_docID][p];
			}
			else if(y[v_index]<y[q_docID])
			{
				feature_node_Minus(n, x[q_docID], x[v_index], minusResult);
				phi_v_index_plus = minusResult;
				effi1 = dotProduct(n, w, phi_v_index_plus);
				effi2 = dotProduct(n, phi_v_index_plus, Y);
				int q;
				for( q=0;q<col_size;q++)
					v_grad_pp[batch_count][q] += (effi1 - C*effi2) * v_pp[q_docID][q];
			}
		}
		
		update_indexes[batch_count++] = v_index;		
	}
	for(int i=0;i<batch_size;i++)
	{
		for(int j=0;j<col_size;j++)
		{
			v_pp[update_indexes[i]][j] -= eta * v_grad_pp[i][j];
		}
	}

	///release variables
	free(Y);//double * Y = Malloc(double,prob->n);
	free(batch_v_grad);//double * batch_v_grad = Malloc(double,batch_size*col_size)
	free(v_grad_pp);//double ** v_grad_pp = Malloc(double*,batch_size);//to store the gradient of batch_size lines of matrix V
	free(update_indexes);//int * update_indexes = Malloc(int,batch_size)// to store the doc 
	free(minusResult);//double* minusResult = Malloc(double,n);

}
void y_avltree_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *y=prob->y;
	double *wa = new double[l];
	avl *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);
	for (i=0;i<nr_subset;i++)
	{
		T=new avl(count[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new avl(count[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(y[pi[i][j].id]);
		}
		delete T;
	}
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}



// To support weights for instances, use GETI(i) (i)

// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
		const problem *prob, double *w, const parameter *param,
		int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init;
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;
		feature_node *xi = prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += beta[i]*val;
			xi++;
		}

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node *xi = prob->x[i];
			while(xi->index != -1)
			{
				int ind = xi->index-1;
				double val = xi->value;
				G += val*w[ind];
				xi++;
			}

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				xi = prob->x[i];
				while(xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: MAX ITERATION REACHED\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void group_queries(const problem *prob, int *nr_subset_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;//l: number of all instances of input file
	int max_nr_subset = 16;
	int nr_subset = 0;
	int *query = Malloc(int,max_nr_subset);//query: distinct query array
	int *count = Malloc(int,max_nr_subset);//count: 
	int *data_query = Malloc(int,l);//data_query: stores each instance's query
	int i;

	for(i=0;i<l;i++)//iterate all instances of input file
	{
		int this_query = (int)prob->query[i];
		int j;
		for(j=0;j<nr_subset;j++)//iterate current distinct queries
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;//data_query[i]: the ith instance's query group label
		if(j == nr_subset)// the ith instace's query doesn't belong to current query group
		{
			if(nr_subset == max_nr_subset)// the query group volumn has reached its upper bound
			{
				max_nr_subset *= 2;
				query = (int *)realloc(query,max_nr_subset*sizeof(int));
				count = (int *)realloc(count,max_nr_subset*sizeof(int));
			}
			query[nr_subset] = this_query;
			count[nr_subset] = 1;
			++nr_subset;
		}
	}////iterate all instances of input file

	int *start = Malloc(int,nr_subset);//start[i]:the i-th query subset's first instance line number in the input file
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)// perm: indices to the original data
	{	
		// the next two sentences equivalent to perm[i]=i;
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	//reset start array after the assignment to perm array
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_subset_ret = nr_subset;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}
double dotProduct(int n, double* v1, double* v2)
{
	int i;
	double f = 0;
	for(i = 0;i<n;i++)
		f += v1[i] * v2[i];
	return f;
}
void feature_node_Minus(int n, feature_node*v1, feature_node* v2, double* minusResult)
{
	int i;
	for(i=0; i<n; i++)//n: feature number
		minusResult[i] = v1[i].value - v2[i].value;
}
void v_pp_to_w(double* w, double**v_pp, int*nr_subset,int*start,int*count,int*perm, double *y,parameter * param,const problem *prob)
{
	int i,j,k,m;
	int big_index,small_index,tmp;
	double alpha_jk;
	double *phi_jk_pt = Malloc(double,prob->n);
	int nr_query = *nr_subset;
	for(i=0;i<prob->n;i++)
	{
		w[i] = 0;
	}
	for(i=0;i<nr_query;i++)
	{
		for(j=0; j<count[i]-1; j++)//iterate all possible pairs within query-i
			for(k=j+1;k<count[i];k++)
			{
				if(y[perm[start[i]+j]]!=y[perm[start[i]+k]])//two instances,under the same query with different labels
				{
					big_index = perm[start[i]+j];
					small_index = perm[start[i]+k];
					if(y[big_index]<y[small_index])//consider preference pair
					{
						tmp = big_index;
						big_index = small_index;
						small_index = tmp;	
					}
					alpha_jk = dotProduct(param->col_size,v_pp[big_index],v_pp[small_index]);
					feature_node_Minus(prob->n,prob->x[big_index],prob->x[small_index],phi_jk_pt);//the minus result is stored in phi_jk_pt
					for(m=0;m<prob->n;m++)
					{
						w[m] += alpha_jk * phi_jk_pt[m];
					}
				}
			}
	}
}
//static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn, int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL)
//{
//	double eps=param->eps;
//	int pos = 0;
//	int neg = 0;
//	clock_t begin,end;
//	for(int i=0;i<prob->l;i++)
//		if(prob->y[i] > 0)
//			pos++;
//	neg = prob->l - pos;
//
//	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;
////	class function
////{
////public:
////	virtual double fun(double *w) = 0 ;
////	virtual void grad(double *w, double *g) = 0 ;
////	virtual void Hv(double *s, double *Hs) = 0 ;
////
////	virtual int get_nr_variable(void) = 0 ;
////	virtual ~function(void){}
////};
//	function *fun_obj=NULL;
//	begin = clock();
//	switch(param->solver_type)
//	{
//		case L2R_L2LOSS_SVC:
//		{
//			double *C = new double[prob->l];
//			for(int i = 0; i < prob->l; i++)
//			{
//				if(prob->y[i] > 0)
//					C[i] = Cp;
//				else
//					C[i] = Cn;
//			}
//			fun_obj=new l2r_l2_svc_fun(prob, C);
//			TRON tron_obj(fun_obj, primal_solver_tol);
//			tron_obj.set_print_string(liblinear_print_string);
//			tron_obj.tron(w);
//			delete fun_obj;
//			delete C;
//			break;
//		}
//		case L2R_L2LOSS_SVR:
//		{
//			double *C = new double[prob->l];
//			for(int i = 0; i < prob->l; i++)
//				C[i] = param->C;
//			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
//			TRON tron_obj(fun_obj, param->eps);
//			tron_obj.set_print_string(liblinear_print_string);
//			tron_obj.tron(w);
//			delete fun_obj;
//			delete C;
//			break;
//		}
//		case Y_RBTREE:
//			{
//				fun_obj=new y_rbtree_rank_fun(prob, param->C, nr_subset, perm, start, count);
//				TRON tron_obj(fun_obj, param->eps);
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case WX_RBTREE:
//			{
//				fun_obj=new wx_rbtree_rank_fun(prob, param->C, nr_subset, perm, start, count);
//				TRON tron_obj(fun_obj, param->eps);
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case SELECTION_TREE:
//			{
//				fun_obj=new selection_rank_fun(prob, param->C, nr_subset, perm, start, count);
//				TRON tron_obj(fun_obj, param->eps);
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case DIRECT_COUNT:
//			{
//				fun_obj=new direct_count(prob, param->C, nr_subset, perm, start, count);
//				TRON tron_obj(fun_obj, param->eps);
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case AVLTREE:
//			{
//				fun_obj=new y_avltree_rank_fun(prob, param->C, nr_subset, perm, start, count);
//				//TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
//				TRON tron_obj(fun_obj, param->eps);//initialize a object of class TRON
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case AATREE:
//			{
//				fun_obj=new y_aatree_rank_fun(prob, param->C, nr_subset, perm, start, count);
//				TRON tron_obj(fun_obj, param->eps);
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case PRSVMP:
//			{
//				fun_obj=new prsvmp_fun(prob, param->C, nr_subset, perm, start, count);
//				TRON tron_obj(fun_obj, param->eps);
//				tron_obj.set_print_string(liblinear_print_string);
//				tron_obj.tron(w);
//				delete fun_obj;
//				break;
//			}
//		case L2R_L1LOSS_SVR_DUAL:
//			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
//			break;
//		default:
//			fprintf(stderr, "ERROR: unknown solver_type\n");
//			break;
//	}
//	end = clock();
//	info("Training time = %g\n",double(end-begin)/double(CLOCKS_PER_SEC));
//}
static void _train_one(const problem *prob, const parameter *param, model *model_, double Cp, double Cn, int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL)
{
	
	clock_t begin,end;

	function *fun_obj=NULL;
	begin = clock();
	switch(param->solver_type)
	{
		
		case AVLTREE:
			{
				fun_obj=new y_avltree_rank_fun(prob, param->C, nr_subset, perm, start, count);
				//TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
				TRON tron_obj(fun_obj, param->eps);//initialize a object of class TRON
				tron_obj.set_print_string(liblinear_print_string);
				//tron_obj.tron(model_->w);
				tron_obj._tron(model_ , prob);
				delete fun_obj;
				break;
			}
	
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
	end = clock();
	info("Training time = %g\n",double(end-begin)/double(CLOCKS_PER_SEC));
}
//
// Interface functions
//
//model* train(const problem *prob, const parameter *param)
//{
//	int i,j;
//	int l = prob->l;
//	int n = prob->n;
//	int w_size = prob->n;
////	struct model
////{
////	struct parameter param;
////	int nr_class;		/* number of classes */
////	int nr_feature;
////	double *w;
////	int *label;		/* label of each class */
////};
//	model *model_ = Malloc(model,1);
//
//	model_->nr_feature=n;
//	model_->param = *param;
//
//	if(param->solver_type == L2R_L2LOSS_SVR ||
//			param->solver_type == L2R_L1LOSS_SVR_DUAL)
//	{
//		model_->w = Malloc(double, w_size);
//		model_->nr_class = 2;
//		model_->label = NULL;
//		train_one(prob, param, &model_->w[0], 0, 0);
//	}
//	else if(param->solver_type == WX_RBTREE||
//			param->solver_type == Y_RBTREE||
//			param->solver_type == SELECTION_TREE||
//			param->solver_type == AVLTREE||
//			param->solver_type == AATREE||
//			param->solver_type == DIRECT_COUNT||
//			param->solver_type == PRSVMP)
//	{
//		model_->w = Malloc(double, w_size);
//		model_->nr_class = 2;
//		model_->label = NULL;
//		int nr_subset;
//		int *start = NULL;
//		int *count = NULL;
//		int *perm = Malloc(int,l);
//		//group_queries(): this function groups instances of the same query into one group,
//		//nr_subset: the query group number
//		//start:a array stores each group's first instance line number(start from 0)
//		//count:a array stores the instances number of each group 
//		//the above four variables' value are assigned during the function group_queries()
//		group_queries(prob, &nr_subset ,&start, &count, perm);
//		//void train_one(const problem *prob, const parameter *param, double *w, 
//		//double Cp, double Cn, int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL)
//		
//		train_one(prob, param, &model_->w[0],0,0, nr_subset, perm, start, count);
//		free(start);
//		free(count);
//		free(perm);
//	}
//	else
//	{
//		int nr_class;
//		int *label = NULL;
//		int *start = NULL;
//		int *count = NULL;
//		int *perm = Malloc(int,l);
//
//		// group training data of the same class
//		group_classes(prob,&nr_class,&label,&start,&count,perm);
//
//		model_->nr_class=nr_class;
//		model_->label = Malloc(int,nr_class);
//		for(i=0;i<nr_class;i++)
//			model_->label[i] = label[i];
//
//		// calculate weighted C
//		double *weighted_C = Malloc(double, nr_class);
//		for(i=0;i<nr_class;i++)
//			weighted_C[i] = param->C;
//		for(i=0;i<param->nr_weight;i++)
//		{
//			for(j=0;j<nr_class;j++)
//				if(param->weight_label[i] == label[j])
//					break;
//			if(j == nr_class)
//				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
//			else
//				weighted_C[j] *= param->weight[i];
//		}
//
//		// constructing the subproblem
//		feature_node **x = Malloc(feature_node *,l);
//		for(i=0;i<l;i++)
//			x[i] = prob->x[perm[i]];
//
//		int k;
//		problem sub_prob;
//		sub_prob.l = l;
//		sub_prob.n = n;
//		sub_prob.x = Malloc(feature_node *,sub_prob.l);
//		sub_prob.y = Malloc(double,sub_prob.l);
//
//		for(k=0; k<sub_prob.l; k++)
//			sub_prob.x[k] = x[k];
//
//		if(nr_class == 2)
//		{
//			model_->w=Malloc(double, w_size);
//
//			int e0 = start[0]+count[0];
//			k=0;
//			for(; k<e0; k++)
//				sub_prob.y[k] = +1;
//			for(; k<sub_prob.l; k++)
//				sub_prob.y[k] = -1;
//
//			train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
//		}
//		else
//		{
//			model_->w=Malloc(double, w_size*nr_class);
//			double *w=Malloc(double, w_size);
//			for(i=0;i<nr_class;i++)
//			{
//				int si = start[i];
//				int ei = si+count[i];
//
//				k=0;
//				for(; k<si; k++)
//					sub_prob.y[k] = -1;
//				for(; k<ei; k++)
//					sub_prob.y[k] = +1;
//				for(; k<sub_prob.l; k++)
//					sub_prob.y[k] = -1;
//
//				train_one(&sub_prob, param, w, weighted_C[i], param->C);
//
//				for(int j=0;j<w_size;j++)
//					model_->w[j*nr_class+i] = w[j];
//			}
//			free(w);
//		}
//
//		free(x);
//		free(label);
//		free(start);
//		free(count);
//		free(perm);
//		free(sub_prob.x);
//		free(sub_prob.y);
//		free(weighted_C);
//	}
//	return model_;
//}
model* _train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
		

	model *model_ = Malloc(model,1);

	model_->nr_feature=n;
	model_->param = *param;

	if(param->solver_type == L2R_L2LOSS_SVR ||
			param->solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		model_->w = Malloc(double, w_size);
		model_->nr_class = 2;
		model_->label = NULL;
		//train_one(prob, param, &model_->w[0], 0, 0);
	}
	else if(param->solver_type == WX_RBTREE||
			param->solver_type == Y_RBTREE||
			param->solver_type == SELECTION_TREE||
			param->solver_type == AVLTREE||
			param->solver_type == AATREE||
			param->solver_type == DIRECT_COUNT||
			param->solver_type == PRSVMP)
	{
		model_->w = Malloc(double, w_size);
		for(i=0;i<w_size;i++)
			model_->w[i] = 0;
		model_->nr_class = 2;
		model_->label = NULL;
		int nr_subset;//nr_subset: the query group number,ie. the total number of distict queries
		int *start = NULL;//start:a array stores each group's first instance line number(start from 0)
		int *count = NULL;//count:a array stores the instances number of each group 
		int *perm = Malloc(int,l);//perm: indices to the original data, ie.  perm[i]=i;(i=0,1,...,l)		
		//the above four variables' value are assigned during the function group_queries()
		group_queries(prob, &nr_subset ,&start, &count, perm);//group_queries(): this function groups instances of the same query into one group,
		//void train_one(const problem *prob, const parameter *param, double *w, 
		//double Cp, double Cn, int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL)
		v_pp_to_w(model_->w, model_->param.v_pp,&nr_subset,start,count,perm,prob->y,&model_->param,prob);
		//train_one(prob, param, &model_->w[0],0,0, nr_subset, perm, start, count);
		_train_one(prob, param, model_,0,0, nr_subset, perm, start, count);
		free(start);
		free(count);
		free(perm);
	}	
	return model_;
}

static void group_queries(const int *query_id, int l, int *nr_query_ret, int **start_ret, int **count_ret, int *perm)
{
	int max_nr_query = 16;
	int nr_query = 0;
	int *query = Malloc(int,max_nr_query);
	int *count = Malloc(int,max_nr_query);
	int *data_query = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_query = (int)query_id[i];
		int j;
		for(j=0;j<nr_query;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_query)
		{
			if(nr_query == max_nr_query)
			{
				max_nr_query *= 2;
				query = (int *)realloc(query,max_nr_query * sizeof(int));
				count = (int *)realloc(count,max_nr_query * sizeof(int));
			}
			query[nr_query] = this_query;
			count[nr_query] = 1;
			++nr_query;
		}
	}

	int *start = Malloc(int,nr_query);
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];

	*nr_query_ret = nr_query;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}

void eval_list(double *label, double *target, int *query, int l, double *result_ret)
{
	int q,i,j,k;
	int nr_query;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int, l);
	id_and_value *order_perm;
	int true_query;
	int ndcg_size;
	long long totalnc = 0, totalnd = 0;
	long long nc = 0;
	long long nd = 0;
	double tmp;
	double accuracy = 0;
	int *l_plus;
	int *int_y;
	int same_y = 0;
	double *ideal_dcg;
	double *dcg;
	double meanndcg = 0;
	double ndcg;
	double dcg_yahoo,idcg_yahoo,ndcg_yahoo;
	selectiontree *T;
	group_queries(query, l, &nr_query, &start, &count, perm);
	true_query = nr_query;
	for (q=0;q<nr_query;q++)
	{
		//We use selection trees to compute pairwise accuracy
		nc = 0;
		nd = 0;
		l_plus = new int[count[q]];
		int_y = new int[count[q]];
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		int_y[order_perm[count[q]-1].id] = 1;
		same_y = 0;
		k = 1;
		for(i=count[q]-2;i>=0;i--)
		{
			if (order_perm[i].value != order_perm[i+1].value)
			{
				same_y = 0;
				k++;
			}
			else
				same_y++;
			int_y[order_perm[i].id] = k;
			nc += (count[q]-1 - i - same_y);
		}
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		//total pairs
		T = new selectiontree(k);
		j = count[q] - 1;
		for (i=count[q] - 1;i>=0;i--)
		{
			while (j>=0 && ( order_perm[j].value < order_perm[i].value))
			{
				T->insert_node(int_y[order_perm[j].id], tmp);
				j--;
			}
			T->count_larger(int_y[order_perm[i].id], &l_plus[order_perm[i].id], &tmp);
		}
		delete T;

		for (i=0;i<count[q];i++)
			nd += l_plus[i];
		nc -= nd;
		if (nc != 0 || nd != 0)
			accuracy += double(nc)/double(nc+nd);
		else
			true_query--;
		totalnc += nc;
		totalnd += nd;
		delete[] l_plus;
		delete[] int_y;
		delete[] order_perm;
	}
	result_ret[0] = (double)totalnc/(double)(totalnc+totalnd);
	for (q=0;q<nr_query;q++)
	{
		ndcg_size = min(10,count[q]);
		ideal_dcg = new double[count[q]];
		dcg = new double[count[q]];
		ndcg = 0;
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		ideal_dcg[0] = pow(2.0,order_perm[0].value) - 1;
		idcg_yahoo = pow(2.0, order_perm[0].value) - 1;
		for (i=1;i<count[q];i++)
			ideal_dcg[i] = ideal_dcg[i-1] + (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+1.0);
		for (i=1;i<ndcg_size;i++)
			idcg_yahoo += (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+2.0);
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		dcg[0] = pow(2.0, label[order_perm[0].id]) - 1;
		dcg_yahoo = pow(2.0, label[order_perm[0].id]) - 1;
		for (i=1;i<count[q];i++)
			dcg[i] = dcg[i-1] + (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 1.0);
		for (i=1;i<ndcg_size;i++)
			dcg_yahoo += (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 2.0);
		if (ideal_dcg[0]>0)
			for (i=0;i<count[q];i++)
				ndcg += dcg[i]/ideal_dcg[i];
		else
			ndcg = 0;
		meanndcg += ndcg/count[q];
		delete[] order_perm;
		delete[] ideal_dcg;
		delete[] dcg;
		if (idcg_yahoo > 0)
			ndcg_yahoo += dcg_yahoo/idcg_yahoo;
		else
			ndcg_yahoo += 1;
	}
	meanndcg /= nr_query;
	ndcg_yahoo /= nr_query;
	result_ret[1] = meanndcg;
	result_ret[2] = ndcg_yahoo;
	free(start);
	free(count);
	free(perm);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}
	return dec_values[0];
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

static const char *solver_type_table[]=
{
"L2R_L2LOSS_SVC", "L2R_L2LOSS_SVR", "L2R_L1LOSS_SVR_DUAL", "DIRECT_COUNT","Y_RBTREE","WX_RBTREE","SELECTION_TREE","AVLTREE","AATREE","PRSVMP",NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_L2LOSS_SVC
			&& param->solver_type != L2R_L2LOSS_SVR
			&& param->solver_type != L2R_L1LOSS_SVR_DUAL
			&& param->solver_type != Y_RBTREE
			&& param->solver_type != WX_RBTREE
			&& param->solver_type != SELECTION_TREE
			&& param->solver_type != AVLTREE
			&& param->solver_type != AATREE
			&& param->solver_type != DIRECT_COUNT
			&& param->solver_type != PRSVMP)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

