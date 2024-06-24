#include <math.h>
#include "my_complex.h"

my_complex cplus(my_complex a, my_complex b) {
  a.x += b.x;
  a.y += b.y;
  return(a);
}

my_complex cmltp(my_complex a, my_complex b) {
  my_complex c;
  c.x = a.x*b.x - a.y*b.y;
  c.y = a.x*b.y + a.y*b.x;
  return(c);
}

my_complex cngtv(my_complex a) {
  a.x = -a.x;
  a.y = -a.y;
  return(a);
}

my_complex cinvs(my_complex a) {
  my_complex dmltp(float, my_complex);
  my_complex conjg(my_complex a);
  return(dmltp(1./(a.x*a.x+a.y*a.y), conjg(a)));
}

my_complex conjg(my_complex a) {
  a.y = -a.y;
  return(a);
}

my_complex dmltp(float a, my_complex b) {
  b.x *= a;
  b.y *= a;
  return(b);
}

my_complex Csqrt(my_complex a) {
  double mo, ar;
  double ccabs(my_complex);
  mo = sqrt(ccabs(a));
  ar = 0.5*atan2(a.y, a.x);
  a.x = mo*cos(ar);
  a.y = mo*sin(ar);
  return(a);
}

my_complex cmplx(float x, float y) {
  my_complex a;
  a.x = x;
  a.y = y;
  return(a);
}

my_complex cphase(my_complex w) {
  double mo;
  mo = exp(w.x);
  return cmplx(mo*cos(w.y), mo*sin(w.y));
}

double ccabs(my_complex a) {
  return(sqrt(a.x*a.x+a.y*a.y));
}
