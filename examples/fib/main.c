#include <stdio.h>
#include <stdlib.h>

int foo(int x, int y) {
  int w = y+3;
  int z = x-50;
  if (z == 47) {
    z = z*z;
    printf("%d", z);
    return z;
  } else {
    w = w*w;
    printf("%d", w);
    return w;
  }
}

int main(int argc, char* argv[]){
  int a = foo(argv[1][0], argv[2][0]);
  int b = foo(argv[2][0], argv[1][0]);
  int c = foo(argv[2][1], argv[1][0]);
  printf(" %d  %d  %d\n",a, b, c);
  return 0;
}
