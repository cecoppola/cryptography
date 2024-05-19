CC		:= gcc -g
CCFLAGS		:= -std=gnu11
LIBS		:= -lgsl -lgslcblas -lm -fopenmp -lgmp

%: %.c
	@$(CC) $(CCFLAGS) $< -o $@ $(LIBS)

