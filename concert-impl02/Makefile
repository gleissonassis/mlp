CFLAGS= -Wall -m64 -g -w -DIL_STD -std=c++11
CXX=g++
HOME=/Users/gleissonassis
ILOG= /Users/gleissonassis/Applications/IBM/ILOG/CPLEX_Studio1271
CPPFLAGS= -DIL_STD -I$(ILOG)/cplex/include -I$(ILOG)/concert/include
CPLEXLIB=-L$(ILOG)/cplex/lib/x86-64_osx/static_pic -lilocplex -lcplex -L$(ILOG)/concert/lib/x86-64_osx/static_pic -lconcert -lm -lpthread
comp:
	$(CXX) $(CFLAGS) $(CPPFLAGS) -o mlp  mlp.cpp   $(CPLEXLIB)
