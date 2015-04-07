#ifndef _TRON_H
#define _TRON_H
#include "linear.h"
class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
	virtual double _fun(double *w){return 0;}
	virtual void _grad(model * model_,const problem* prob,int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();

	void tron(double *w);
	void _tron(model * model_,const problem * prob);	
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
