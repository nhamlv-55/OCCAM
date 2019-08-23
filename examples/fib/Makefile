CC=gclang
all: main

main: main.c 
	${CC} -mllvm -disable-llvm-optzns -O1 -Wall  main.c -o main 
	get-bc main

clean:
	rm -f *~ .*.bc *.bc *.ll .*.o main main_slash
	rm -rf slash
