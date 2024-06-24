/*********************************************************
*			my_complex.h
*  header file for my_complex data type, source codes are in
*  my_complex.c and fft.c
*********************************************************/

#ifndef __MY_COMPLEX__
  #define __MY_COMPLEX__


/* data type */
typedef struct { float x; float y;} my_complex;

/* constants */
#define PI	3.1415926
#define IMAGE cmplx(0., 1.)
#define One  cmplx(1., 0.)
#define Zero cmplx(0., 0.)

/* basic operations */
my_complex cplus(my_complex a, my_complex b);
my_complex cmltp(my_complex a, my_complex b);
my_complex cngtv(my_complex a);
my_complex cinvs(my_complex a);
my_complex conjg(my_complex a);
my_complex dmltp(float a, my_complex b);
my_complex Csqrt(my_complex a);
my_complex cmplx(float x, float y);
my_complex cphase(my_complex w);
double  ccabs(my_complex a);

/* fft */
void    fft(my_complex *a, int n, float dt);	/* dt>0: forw.; dt<0: inv */
void    fftr(my_complex *x, int n, float dt);

/* convolution and correlation */
void	cor(my_complex *a, my_complex *b, float dt, int nft);
void	conv(float *, int, float *, int);
float	*crscrl(int,float *,float *,int);
float	maxCor(float *, float *, int, int, int *, float *);
float	maxCorSlide(float *, float *, int, int, float, int *, float *);

/* integration, moving average, and differentiation */
float amp(float t1, float t2, float *data, int n);
float acc(float *data, int n, float t1, float t2, int half);
void cumsum(float *a, int n, float dt);
void maver(float *a, int n, int m);	/* m-point moving average */
void diffrt(float *a, int n, float dt);
void sqr(float *a, int n);
void revers(float *a, int n);

/* windowing */
float *coswndw(int, float);

/* high-pass filtering */
void	filter(my_complex *, int, float, float, float, int);

/* find max. values in an array, return the shift */
int findMax(float *a, int n, float *amp);

/* find max. absolute values in an array, return the shift */
int findMaxAbs(float *a, int n, float *amp);

/* remove trend a+b*x */
void rtrend(float *, int);

/* some operation on spectrum */
void fltGauss(my_complex *u, int n, float gauss);
void shiftSpec(my_complex *u, int n, float shift);
void specAdd(my_complex *a, my_complex *b, int n);
void specMul(my_complex *a, my_complex *b, int n);
void specScale(my_complex *a, float c, int n);
float specPwr(my_complex *u, int n);

#endif
