#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

struct id_and_value
{
	int id;
	double value;
};

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	int *query;
	double *y;
	struct feature_node **x;
};

enum {L2R_L2LOSS_SVC, L2R_L2LOSS_SVR, L2R_L1LOSS_SVR_DUAL, DIRECT_COUNT,Y_RBTREE,WX_RBTREE,SELECTION_TREE,AVLTREE,AATREE,PRSVMP  }; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
	double ** v_pp;
	int ite;
	int col_size;
	int batch_size;
	double eta;
};
struct _parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
	double ** v_pp;
	int ite;
	int col_size;
};
struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
};

struct model* train(const struct problem *prob, const struct parameter *param);
struct model* _train(const struct problem *prob, const struct parameter *param);
double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

double dotProduct(int n, double* v1, double* v2);
void feature_node_Minus(int n, feature_node*v1, feature_node* v2, double* minusResult);
void v_pp_to_w(double* w, double**v_pp, int*nr_subset,int*start,int*count,int*perm, double *y,parameter * param,const problem *prob);
void group_queries(const problem *prob, int *nr_subset_ret, int **start_ret, int **count_ret, int *perm);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
void set_print_string_function(void (*print_func) (const char*));
void eval_list(double *label, double *target, int *query, int l, double *result_ret);
#ifdef FIGURE56
void evaluate_test(double* w);
extern struct feature_node *x_spacetest;
extern struct problem probtest;
#endif

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

