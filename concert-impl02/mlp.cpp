#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <ilcplex/ilocplex.h>
#include <algorithm>
#include <map>
#include <sys/timeb.h>


using namespace std;
#define EPSILON 10e-6
#define TimeLimit 200
#define M 10e20

ILOSTLBEGIN

typedef IloArray<IloNumVarArray> NumVarMatrix2;
typedef IloArray<NumVarMatrix2> NumVarMatrix3;

/// ==============================================
/// Benders decompositon class
/// ==============================================

class Mono{
	public:
		double time_begin;

		/// General parameter
		int mip_time_limit;
		double ub;
		double lb;
		double gap;
		int frac_x;
		bool relaxed;
		bool initial_solution;

		/// Model parameter
    int n;
    vector<vector<int>>c;

		/// Model parameter
    IloEnv env;
    IloModel mod;
    IloCplex cplex;
    IloObjective fo;
    NumVarMatrix2 vx;
    NumVarMatrix2 vf;
    NumVarMatrix2 vp;
    vector<vector<double>> x_bar;
    vector<vector<double>> f_bar;
    vector<vector<int>> x_til;
    vector<vector<double>> f_til;
    vector<vector<int>> x_best;
    vector<vector<double>> f_best;
    vector<vector<double>> x;
    vector<vector<double>> f;
    vector<vector<IloConversion>> convx;
    vector<vector<IloConversion>> convf;
	public:
		Mono(){
			ub = M;
			lb = 0.0;
			gap = 1;
			initial_solution = false;
			mod = IloModel(env);
			cplex = IloCplex(mod);
			frac_x = 0;
		}
		void Read_data(string);	// Função para leitura dos dados
		void Create_model();   // Criar um modelo monolítico
		void GetRelaxedSolution(); // Guardar solução do relaxado para variável bar
		void GetMIPSolution(); // Guardar solução do relaxado para variável til
		bool Solve_mip_model();  // Resolve o problema MIP
		bool Solve_relaxed_model();  // Resolve o problema relaxado
		bool RENS();   // Heuristica construtiva RENS
		bool FP(int T);   // Heuristica construtiva FP
		bool FP2();   // Heuristica construtiva FP
		void FlipFP(int T);   // Heuristica construtiva FP
		void LocalSearch();   // Heuristica construtiva FP
		void LocalBranching(int k0);   // Heurística Local Branching
		void VNDB(int k0);   //Heurística VNDB
		void Cria_problema_reduzido(vector<vector<double> >);  // Resolve o problema MIP
		void Reinitialize_bounds();
		void Convergent_heuristic();   // Heurística convergente
		void Print_results(); // Imprime resultados
		void Display_relaxed_configuration(); // Display configurações
		void Display_mip_configuration();// Display configurações
		void Display_best_configuration();// Display configurações
		void Display_configuration();// Display configurações
		float ObjectiveValue(vector<vector<double>>);
    float ObjectiveValue(vector<vector<int>>);
};


uint64_t system_current_time_millis();
/// ==============================================
/// main program
/// ==============================================
int max(int a, int b);
int min(int a, int b);
int GeneratedCandidate(int LCT);
double Rand();
bool igual(double x, double y);

int main (int argc, char *argv[]){

	Mono *mono = new Mono();

	try{


		///===========================
		/// Inicializações
		///==========================
		string arquivo(argc == 1 ? "/Users/gleissonassis/Documents/github/mlp/concert-impl02/problems/15_1_100_1000.txt" : argv[1]);
		mono->Read_data(arquivo); /// Leitura do arquivo de dados
		int constr_heur = (argc > 2)? atof(argv[2]): 0;
		int refin_heur = (argc > 3)? atof(argv[3]): 1;
		int k0 = (argc > 4)? atof(argv[4]): 8;
		int limit = (argc > 5)? atof(argv[5]): 200;
		int T = round(0.05 * mono-> n * mono->n);

		IloTimer crono(mono->env);  /// Variável para coletar o tempo
		mono->mip_time_limit = limit;

		///===========================
		/// criando o modelo principal
		///==========================

		mono->Create_model();

		/// ==========================
		/// cplex basic configurations
		/// ==========================

		mono->cplex.setParam(IloCplex::EpGap, 0.00001);
		mono->cplex.setWarning(mono->env.getNullStream());
		//mono->cplex.setOut(mono->env.getNullStream());
		//mono->cplex.setParam(IloCplex::Threads,1);
		mono->cplex.setParam(IloCplex::PreInd, false);

		///==============================
		/// Resolvendo o problema
		///==============================
		srand(10);
		crono.start();
		clock_t begin = clock();
		mono->time_begin = crono.getTime();
		uint64_t s = system_current_time_millis();
		switch(constr_heur){
			case 0:
				cout << "Solving MIP Model" << endl;;
				mono->Solve_mip_model();
				break;
			case 1:
				cout << "Rens " << endl;
				mono->RENS();
				break;
			case 2:
				cout << "FP " <<endl;
				mono->FP(T);
				break;
			case 3:
				cout << "FP " << endl;
				mono->FP2();
				break;
			case 4:
				cout << "MIP " << endl;
				mono->Solve_mip_model();
				break;
			default:
				cout << "Invalid options: using MIP Model"<<endl;
				mono->Solve_mip_model();
				break;

		}

		if (constr_heur != 0) {
			//time_begin = crono.getTime();
			switch(refin_heur){
				case 1:
					cout <<"Local branching " << endl;
					mono->LocalBranching(k0);
					break;
				case 2:
					cout << "VNDB " <<endl;
					mono->VNDB(k0);
					break;
				case 3:
					cout << "Convergent heuristic " <<endl;
					mono->Convergent_heuristic();
					break;
				default:
					cout << "Opção inválidade: usando LB" <<endl;
					mono->LocalBranching(k0);
					break;
			}
		} else {
			cout << " The MIP was solved so no improment heuristic will be applied to solution" << endl;
		}

		//fprintf(fp,"%d\t%1.4f\n",  (int)mono->ub, (double) crono.getTime() - time_begin);

		crono.stop();
		clock_t end = clock();
		uint64_t e = system_current_time_millis();

		///=====================================
		/// Salvando os resultados
		///=====================================
		/// Pegar gap de relaxação linear, gap da solução encontrada, tempo de resolução, número de iterações, fração de variáveis fracionárias
		FILE *fp;
		printf("%s\t%d\t%1.2f\t%1.2f\t%1.2f\t%1.6f\t%1.1f\n",  argv[1], mono->n,   100 *  mono->gap,  mono->lb, mono->ub, (double)(e - s) / 1000, (double) 100 * (mono->frac_x)/(mono->n * mono->n));
		fp = fopen("results.txt","aw+");
		fprintf(fp,"%s;%d;%d;%d;%d;%1.2f;%1.2f;%1.2f;%1.6f;%1.1f\n",
			argv[1],
			mono->n,
			constr_heur,
			refin_heur,
			k0,
			100 *  mono->gap,
			mono->lb,
			mono->ub,
			(double)(e - s) / 1000,
			(double) 100 * (mono->frac_x)/(mono->n * mono->n));
		fclose(fp);
	}
	catch (IloException& ex) {
		cerr << "Error: " << ex << endl;
	}
	return 0;

}

uint64_t system_current_time_millis()
{
#if defined(_WIN32) || defined(_WIN64)
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    return (uint64_t)(((timebuffer.time * 1000) + timebuffer.millitm));
#else
    struct timeb timebuffer;
    ftime(&timebuffer);
    return (uint64_t)(((timebuffer.time * 1000) + timebuffer.millitm));
#endif
}

int max(int a, int b){
	int x = a > b ? a:b;
	return (x);
}
int min(int a, int b){
	int x = a < b ? a:b;
	return (x);
}

int GeneratedCandidate(int LCT){
	return ( (int) (rand() % LCT));
}

double Rand(){
	return ( (double) rand() / RAND_MAX);
}

bool igual(double x, double y){
	if(x - y <=  EPSILON && y - x <= EPSILON){
		return true;
	}
	else{
		return false;
	}
}
std::vector<int> make_vector(int a, int b) {
  std::vector<int> result;
  result.push_back(a);
  result.push_back(b);
  return result;
}

float Mono::ObjectiveValue(vector<vector<int>> f){
    double objvalue = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
					if(i != j) {
            objvalue += f[i][j] * c[i][j];
					}
        }
    }
    return objvalue;
}


float Mono::ObjectiveValue(vector<vector<double>> f){
    double objvalue = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            objvalue += f[i][j] * c[i][j];
        }
    }
    return objvalue;
}

void  Mono::Read_data(string name){
	cout << "----------------------------------------" << endl;
  cout << "Starting Read_data method" << endl;

  ifstream arq(name);
  if (!arq.is_open()){
    cout << "Error openning file: " << name << endl;
    arq.close();
    exit(EXIT_FAILURE);
  }
  arq >> n;

  x = vector<vector<double>>(n, vector<double>(n));
  c = vector< vector<int> >(n, vector<int>(n));

  for (int i = 0; i <  n; i++) {
    for (int j = 0; j <  n; j++) {
      arq >> c[i][j];
    }
  }
  arq.close();

  cout << "Matrix:" << endl << endl;

  for(int i = 0; i < n; i++) {
    for(int j=0; j< n ; j++) {
      cout << c[i][j] << " ";
    }
    cout << endl;
  }

  cout << endl;
  cout << "Read_data method executed successfully" << endl;
  cout << "---------------------------------------" << endl;
}

void Mono::Create_model(){
	//========================
  // Declaração do modelo
  //========================


  /*cplex.setParam(IloCplex::EpGap, 0.00001);
  cplex.setParam(IloCplex::TiLim, 36000);
  cplex.setWarning(env.getNullStream());
  //cplex.setOut(env.getNullStream());
  */


  vx = IloArray<IloNumVarArray>(env, n);
  for(int i = 0; i < n; i++){
    vx[i] = IloNumVarArray(env, n, 0, 1, ILOBOOL);
  }

  vp = IloArray<IloNumVarArray>(env, n);
  for(int i = 0; i < n; i++){
      vp[i] = IloNumVarArray(env, n, 0, 1, ILOFLOAT);
  }

  vf = IloArray<IloNumVarArray>(env, n);
  for(int i = 0; i < n; i++){
      vf[i] = IloNumVarArray(env, n, 0, IloInfinity, ILOFLOAT);
  }

  x_bar = vector<vector<double>>(n, vector<double>(n));
  f_bar = vector<vector<double>>(n, vector<double>(n));
  x_til = vector<vector<int>>(n, vector<int>(n));
  f_til = vector<vector<double>>(n, vector<double>(n));
  x_best = vector<vector<int>>(n, vector<int>(n));
  f_best = vector<vector<double>>(n, vector<double>(n));
  relaxed = false;

  convx = vector<vector<IloConversion>>(n, vector<IloConversion>(n));
  //convf = vector<vector<IloConversion>>(n, vector<IloConversion>(n));
  for (int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      convx[i][j] = IloConversion(env, vx[i][j], ILOFLOAT);
      //convf[i][j] = IloConversion(env, vf[i][j], ILOFLOAT);
    }
  }

  // ==============================
  // objective function - Master
  // ==============================

  IloExpr expfo(env);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++) {
      expfo += c[i][j] * vf[i][j];
    }
  }
  fo = IloAdd(mod, IloMinimize(env, expfo));
  expfo.end();


  //RESTRIÇÕES
  for(int j = 0; j < n; j++){
    IloExpr r1(env);
    for(int i = 0; i < n; i++){
      if(i != j) {
        r1 += vx[i][j];
      }
    }
    mod.add(r1 == 1);
    r1.end();
  }

  for(int i = 0; i < n; i++){
    IloExpr r2(env);
    for (int j = 0; j < n; j++){
      if(i != j) {
        r2 += vx[i][j];
      }
    }
    mod.add(r2 == 1);
    r2.end();
  }

  IloExpr r3(env);
  for(int j = 1; j < n; j++){
    r3 += vf[0][j];
  }
  mod.add(r3 == n);
  r3.end();

  for(int k = 1; k < n; k++){
    IloExpr r4(env);
    for(int i = 0; i < n; i++){
      r4 += vf[i][k];
    }
    for(int j = 0; j < n; j++){
      r4 -= vf[k][j];
    }
    mod.add(r4 == 1);
    r4.end();
  }

  IloExpr r5(env);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      r5 += vf[i][j];
    }
  }
  mod.add(r5 == (n*(n+1))/2);
  r5.end();

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      IloExpr r6(env);
      r6 = vf[i][j];
      mod.add(r6 <= n * vx[i][j]);
      r6.end();
    }
  }
}

void Mono::GetRelaxedSolution(){
	cout << "-----------------------------------------------" << endl;
  cout << "Iniciando execução do método GetRelaxedSolution" << endl;
	for (int j = 0; j < n; j++){
    for (int i = 0; i < n; i++){
      x_bar[i][j] = cplex.getValue(vx[i][j]);
      f_bar[i][j] = cplex.getValue(vf[i][j]);
    }
  }
	cout << "cplex.getObjValue: \t" << cplex.getObjValue() << endl;
	cout << "ObjetiveValue: \t\t" << ObjectiveValue(f_bar) << endl;
	cout << endl;
  cout << "Término da execução do método GetRelaxedSolution" << endl;
	cout << "------------------------------------------------" << endl;
}

void Mono::GetMIPSolution(){
	cout << "-------------------------------------------" << endl;
  cout << "Iniciando execução do método GetMIPSolution" << endl;
	for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      x_til[i][j] = cplex.getValue(vx[i][j]);
      f_til[i][j] = cplex.getValue(vf[i][j]);
    }
  }
	cout << endl;
	cout << "cplex.getObjValue: \t" << cplex.getObjValue() << endl;
	cout << "ObjetiveValue: \t\t" << ObjectiveValue(f_til) << endl;
	cout << endl;
  cout << "Término da execução do método GetMIPSolution" << endl;
	cout << "--------------------------------------------" << endl;
}

bool Mono::Solve_relaxed_model(){
	cout << "------------------------------------------------" << endl;
  cout << "Iniciando execução do método Solve_relaxed_model" << endl;

  if(!relaxed) {
    relaxed = true;

    for (int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        mod.add(convx[i][j]);
        //mod.add(convf[i][j]);
      }
    }
  }

  cplex.solve();

  if(cplex.getStatus() == IloAlgorithm::Optimal){
    GetRelaxedSolution();
    lb = cplex.getObjValue();
    frac_x = 0;

    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        if(x_bar[i][j] >= EPSILON and x_bar[i][j] <= 1- EPSILON){ // Se solução binária
          frac_x++;
        }
      }
    }

    cout << endl << "Solve_relaxed_model x" << endl << endl;

    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        cout << (i == j ? 0 : x_bar[i][j]) << " ";
      }
      cout << endl;
    }

    cout << endl << "Solve_relaxed_model f" << endl << endl;

    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        cout << f_bar[i][j] << " ";
      }
      cout << endl;
    }

    cout << "FO: " << lb << endl;

    cout << endl;
    cout << "Término da execução do método Solve_relaxed_model: true" << endl;
    cout << "-------------------------------------------------------" << endl << endl;
    return true;
  } else {
      cout << endl;
      cout << "Término da execução do método Solve_relaxed_model: false" << endl;
      cout << "--------------------------------------------------------" << endl << endl;
      return false;
  }
}

bool Mono::Solve_mip_model(){
	if(relaxed){
    for (int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        mod.remove(convx[i][j]);
        //mod.remove(convf[i][j]);
      }
    }
    relaxed = false;
  }

	cout << "Setting up the time limit for " << mip_time_limit << "s" << endl;
	cplex.setParam(IloCplex::TiLim, mip_time_limit);

  cplex.solve();

  if(cplex.getStatus() == IloAlgorithm::Optimal){
    ub = min(ub, cplex.getObjValue());
		cout << "Solve_mip_model ub: " << ub << endl;
    GetMIPSolution();
    Display_mip_configuration();
    gap = (ub - lb)/ub;
    return true;
  } else {
		ub = min(ub, cplex.getObjValue());
		cout << "Solve_mip_model ub: " << ub << endl;
    GetMIPSolution();
    Display_mip_configuration();
    gap = (ub - lb)/ub;
    return false;
  }
}

void Mono::Display_relaxed_configuration(){
  cout << "------------------------------------------------------------" << endl;
  cout << "Iniciando a execução do método Display_relaxed_configuration" << endl << endl;

  cout << "Display_relaxed_configuration x" << endl << endl;

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      cout << (i == j ? 0 : x_bar[i][j]) << " ";
    }
    cout << endl;
  }

  cout << endl;
  cout << "Display_relaxed_configuration f" << endl << endl;

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      cout << (i == j ? 0 : f_bar[i][j]) << " ";
    }
    cout << endl;
  }

  cout << "FO: " << ObjectiveValue(f_bar) << endl;

  cout << "Término da execução do método Display_relaxed_configuration" << endl;
  cout << "-----------------------------------------------------------" << endl << endl;
}

void Mono::Display_mip_configuration(){
    cout << "--------------------------------------------------------" << endl;
    cout << "Iniciando a execução do método Display_mip_configuration" << endl << endl;

    cout << "Display_mip_configuration x" << endl << endl;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            cout << (i == j ? 0 : x_til[i][j]) << " ";
        }
        cout << endl;
    }

    cout << endl;
    cout << "Display_mip_configuration f" << endl << endl;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            cout << (i == j ? 0 : f_til[i][j]) << " ";
        }
        cout << endl;
    }

    cout << "FO: " << ObjectiveValue(f_til) << endl;

    cout << "Término da execução do método Display_mip_configuration" << endl;
    cout << "-------------------------------------------------------" << endl << endl;
}

void Mono::Display_configuration(){
	cout << "----------------------------------------------------" << endl;
  cout << "Iniciando a execução do método Display_configuration" << endl << endl;

  cout << "Display_configuration x" << endl << endl;

	for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      cout << (i == j ? 0 : fabs(cplex.getValue(vx[i][j]))) << " ";
    }
    cout << endl;
  }

  cout << endl;
  cout << "Display_configuration f" << endl << endl;

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      cout << (i == j ? 0 :  cplex.getValue(vf[i][j])) << " ";
    }
    cout << endl;
  }

  cout << "FO: " << cplex.getObjValue() << endl;

  cout << "Término da execução do método Display_configuration" << endl;
  cout << "---------------------------------------------------" << endl << endl;
}

bool Mono::FP(int T){

		//===============================
		// Declara vetor de arredondamentos
		//===============================

		double new_hamming_distance = 1000;
		initial_solution = false;
		int itermax = 15; /// Redefinir um valor coerente de acordo com seu problema


		bool solution_status = Solve_relaxed_model();
		int h = 1;
		while((initial_solution == false) && (h <= itermax)){
//		while(initial_solution == false){
			Display_mip_configuration();
			double last_hamming_distance = new_hamming_distance;
			for(int i = 0; i < n; i++){ /// Arredonda para o inteiro mais próximo
				for(int j = 0; j < n; j++){
					x_til[i][j] = round(x_bar[i][j]);
				}
			}

			int sum_til = 0;
			for(int i = 0; i < n; i++){ /// Atualiza a f.o.
				for(int j = 0; j < n; j++){
					fo.setLinearCoef(vx[i][j], 1 - 2 * x_til[i][j]);
					sum_til -= x_til[i][j];
				}
			}
			Solve_relaxed_model();
			new_hamming_distance = cplex.getObjValue();

			cout<<h<<"\t"<<new_hamming_distance<<"\t"<<last_hamming_distance<<"\t"<<sum_til<<"\t"<<new_hamming_distance - sum_til<<"\t"<<igual(new_hamming_distance, -sum_til)<<endl;
			if(igual(new_hamming_distance, sum_til)){ /// Verifica se a distância de hamming igual a zero
				initial_solution = true;
				ub = ObjectiveValue(f_til);
			}
			else if(igual(new_hamming_distance, last_hamming_distance)){/// Indicador de ciclagem
				FlipFP(T);
				Solve_relaxed_model();
			}
			h++;
		}
		for(int i = 0; i < n; i++){ /// Volta coeficientes da f.o. para os valores iniciais
			for(int j = 0; j < n; j++){
				fo.setLinearCoef(vx[i][j], 0.0);
			}
		}
		return initial_solution;
}

void Mono::FlipFP(int T){
	int TT = (0.5 + Rand()) * T;
	map <double, vector<int> > mymap; /// Vetor que guarda os dados ordenados

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(fabs(x_bar[i][j] - x_til[i][j]) >= EPSILON){
			double value = fabs(x_bar[i][j] - x_til[i][j]) + Rand() * 0.001;
				mymap.insert (pair<double, vector<int> >(value, make_vector(i,j)));
			}
		}
	}
	map <double, vector<int> >::iterator it = mymap.end();
	--it;
	TT = min(TT, mymap.size());
	for(int r = 1; r <= TT; r++){
		if(it->second[1] >= 0){ /// Flip x
			x_til[it->second[0]][it->second[1]] =  (x_til[it->second[0]][it->second[1]] + 1) % 2;
			if(mymap.begin()==it){
				break;
			}
			else{
				it--;
			}
		}
	}
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			fo.setLinearCoef(vx[i][j], 1 - 2 * x_til[i][j]);
		}
	}
}

bool Mono::FP2(){
	//===============================
  // Declara vetor de arredondamentos
  //===============================

  double new_hamming_distance = 1000000;
  initial_solution = false;
  int itermax = 15; /// Redefinir um valor coerente de acordo com seu problema


  bool solution_status = Solve_relaxed_model();
  int h = 1;
  while((initial_solution == false) and (h <= itermax)){
    ///		while(initial_solution == false){

    double last_hamming_distance = new_hamming_distance;
    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        x_til[i][j] = round(x_bar[i][j]);
        //f_til[i][j] = round(f_bar[i][j]);
      }
    }
    int sum_til = 0;
    for(int i = 0; i < n; i++){ /// Muda função objetivo
      for(int j = 0; j < n; j++){
        if(i != j) {
          fo.setLinearCoef(vx[i][j], 1 - 2 * x_til[i][j]);
          //fo.setLinearCoef(vf[i][j], 1 - 2 * f_til[i][j]);
          sum_til -= x_til[i][j];
          //sum_til -= f_til[i][j];
        }
      }
    }
    Solve_relaxed_model();
    new_hamming_distance = cplex.getObjValue();

		cout << "FP cplex.getObjValue(): " << cplex.getObjValue() << endl;
		cout << "FP ub: " << ub << endl;
		cout << "FP Diff: " << ub - ObjectiveValue(f_til) << endl;

    cout<<h<<"\t"<<new_hamming_distance<<"\t"<<last_hamming_distance<<"\t"<<sum_til<<"\t"<<new_hamming_distance - sum_til<<"\t"<<igual(new_hamming_distance, -sum_til)<<endl;

    if(igual(new_hamming_distance, sum_til)){ /// Verifica se distância de hamming igual a zero
      initial_solution = true;
      ub = ObjectiveValue(f_til);
    }
    else if(igual(new_hamming_distance, last_hamming_distance)){///Indicador de ciclagem
      LocalSearch(); /// Faz busca local ao invés de flip
      initial_solution = true;
      ub = ObjectiveValue(f_til);
    }
    h++;
  }
  /// Retorna os coeficientes da função objetivo aos valores iniciais
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      fo.setLinearCoef(vx[i][j], 0.0);
      //fo.setLinearCoef(vf[i][j], 0.0);
    }
  }
  return initial_solution; /// x_til e f_til guarda a solução inicial encontrada
}

void Mono::LocalSearch(){

		cplex.setParam(IloCplex::TiLim, 100); //Tempo limite de resolução

	cout << "------------------------------------------" << endl << endl;
  cout << "Iniciando a execução do método LocalSearch" << endl;
  bool criterioparada = false;
  int k = 5; /// Redefinir um valor coerente de acordo com seu problema
  int h = 1;
  while(criterioparada == false){
		cout << "h:" << h << endl;
    IloExpr branch(env);
    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        branch += vx[i][j] * ( 1 - x_til[i][j]) + x_til[i][j] * (1 - vx[i][j]);
        //branch += vf[i][j] * ( 1 - f_til[i][j]) + f_til[i][j] * (1 - vf[i][j]);
      }
    }
    IloExtractable left_branching = mod.add( branch <= k);
    for(int j = 0; j < n; j++){
        for(int i = 0; i < n; i++){
            fo.setLinearCoef(vx[i][j], 0.0);
            //fo.setLinearCoef(vf[i][j], 0.0);
        }
    }
    Solve_mip_model();
    if ( cplex.getStatus() == IloAlgorithm::Optimal or cplex.getStatus() == IloAlgorithm::Feasible){
        GetMIPSolution();
        Display_mip_configuration();
        if(cplex.getObjValue() < ub){
            ub = cplex.getObjValue();
        }
        criterioparada = true;
    }
    else if (cplex.getStatus() == IloAlgorithm::Infeasible){// or ((cplex.getStatus() == IloAlgorithm::Optimal) and (cplex.getObjValue() > ub - EPSILON))){
        mod.add(branch >= k + 1);
        k = k + 1;
    }
    mod.remove(left_branching);
    branch.end();
    h++;

		if(h > n) {
			criterioparada = true;
		}
  }
  for(int j = 0; j < n; j++){ /// Atualiza coeficiente da f.o.
    for(int i = 0; i < n; i++){
      fo.setLinearCoef(vx[i][j], 1 - 2 * x_til[i][j]);
      //fo.setLinearCoef(vf[i][j], 1 - 2 * f_til[i][j]);
    }
  }

  cout << "Término da execução do método LocalSearch" << endl;
  cout << "-----------------------------------------" << endl << endl;
}



bool Mono::RENS(){
  //==========================
	// Resolve e guarda a solução do problema relaxado
	//==========================
	Solve_relaxed_model();
	lb = cplex.getObjValue();
	GetRelaxedSolution();

	Display_relaxed_configuration();

	//=============================
	// Criar o problema reduzido
	//=============================
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			if(x_bar[i][j] <= EPSILON or x_bar[i][j] >= 1- EPSILON){ // Se solução binária
				vx[i][j].setBounds(x_bar[i][j], x_bar[i][j]);
			}
		}
	}

	//============================
	//  Resolve o problema reduzido como mip
	//=========================

	Solve_mip_model();
	cout<<"Mip status "<<cplex.getStatus()<<endl;
	if(cplex.getStatus() == IloAlgorithm::Optimal){
		GetMIPSolution();
		Display_mip_configuration();
		Display_configuration();
		initial_solution = true;
		ub = min(ub, cplex.getObjValue());

	}
	else{
		cout<<"Solução inicial não encontrada"<<endl;
		exit(0);
	}
	//=============================
	// Retornando aos bounds iniciais
	//=============================

	Reinitialize_bounds();
	return initial_solution;
}

void Mono::LocalBranching(int k0){

	cplex.setParam(IloCplex::TiLim, 10); //Tempo limite de resolução

	bool criterioparada = false;
	int k = k0;
	int h = 1;
	int ciclagem = 0;
	while(criterioparada == false){
		cout << "H: " << h << "K: " << k << endl;
		Display_mip_configuration();
		IloExpr branch(env);
		for(int j = 0; j < n; j++){
			//branch += vy[j] * ( 1 - y_til[j]) + y_til[j] * (1- vy[j]);
			for(int i = 0; i < n; i++){
				//branch += vf[i][j] * ( 1 - f_til[i][j]) + f_til[i][j] * (1 - vf[i][j]);
				branch += vx[i][j] * ( 1 - x_til[i][j]) + x_til[i][j] * (1 - vx[i][j]);
			}
		}

		try {
    	IloExtractable left_branching = mod.add(1 <= branch <= k);
	    cplex.solve();

			if(ub - cplex.getObjValue() < EPSILON) {
				ciclagem++;
			} else {
				ciclagem = 0;
			}

			cout << "LocalBranching cplex.getObjValue(): " << cplex.getObjValue() << endl;
			cout << "LocalBranching ub: " << ub << endl;
			cout << "LocalBranching Diff: " << ub - cplex.getObjValue() << endl;
			cout << "LocalBranching ciclagem: " << ciclagem << endl;

			if ( (cplex.getStatus() == IloAlgorithm::Optimal || cplex.getStatus() == IloAlgorithm::Feasible)
			&& (cplex.getObjValue() < ub - EPSILON)){
				GetMIPSolution();
				ub = cplex.getObjValue();
				mod.add(branch >= k + 1);
			}
			else if (cplex.getStatus() == IloAlgorithm::Infeasible || ((cplex.getStatus() == IloAlgorithm::Optimal) && (cplex.getObjValue() > ub - EPSILON))){
	///			k = k * (1.1) + 1; // + max(1, floor(k/2));
				k = k  + max(1, floor(k/2));
			}
			else {
				k = max(1, floor(k/2));
			}
			mod.remove(left_branching);
			branch.end();
	    h++;
		} catch (IloException& ex) {
			criterioparada = true;
			cerr << "Error: " << ex << endl;
		}

    if(k >= n || ciclagem >= 10){
			criterioparada = true;
		}
	}
}

void Mono::VNDB(int k0){

	cplex.setParam(IloCplex::TiLim, 10); //Tempo limite de resolução
	bool criterioparada = false;
	int k = k0;
	int h = 1;
	while (criterioparada == false) {
		IloExpr branch(env);
		for (int j = 0; j < n; j++) {
		//branch += vy[j] * ( 1 - y_til[j]) + y_til[j] * (1- vy[j]);
			for (int i = 0; i < n; i++) {
				branch += vf[i][j] * ( 1 - f_til[i][j]) + f_til[i][j] * (1 - vf[i][j]);
				branch += vx[i][j] * ( 1 - x_til[i][j]) + x_til[i][j] * (1 - vx[i][j]);
			}
		}

		try {
			IloExtractable left_branching = mod.add(1 <= branch <= k);
			cplex.solve();

			cout << "VNDB k: " << k << endl;
			cout << "VNDB h: " << h << endl;
			cout << "VNDB cplex.getObjValue(): " << cplex.getObjValue() << endl;
			cout << "VNDB ub: " << ub << endl;
			cout << "VNDB Diff: " << ub - cplex.getObjValue() << endl;
			cout << "VNDB cplex.getStatus(): " << cplex.getStatus() << endl;

			if (cplex.getStatus() == IloAlgorithm::Optimal){
				GetMIPSolution();
				if(cplex.getObjValue() < ub){
					ub = cplex.getObjValue();
					x_best = x_til;
				}
				mod.add(branch >= k + 1);
				k = k0;
			} else if (cplex.getStatus() == IloAlgorithm::Feasible){
				GetMIPSolution();
				if(cplex.getObjValue() < ub){
					ub = cplex.getObjValue();
					x_best = x_til;
				}
				k = k0;
				mod.add(branch >= 1);
			} else if (cplex.getStatus() == IloAlgorithm::Infeasible){// or ((cplex.getStatus() == IloAlgorithm::Optimal) and (cplex.getObjValue() > ub - EPSILON))){
				mod.add(branch >= k + 1);
				k = k + 1;
			} else {
				criterioparada = true;
			}
			mod.remove(left_branching);
			branch.end();
			h++;
		} catch (IloException& ex) {
			criterioparada = true;
			cerr << "Error: " << ex << endl;
		}

		if (k >= n || h > 10) {
			criterioparada = true;
		}
	}
}

void Mono::Convergent_heuristic(){
	bool criterioparada = false;
	int h= 1;
	double Relaxed_Value = 0.0;
	while(criterioparada == false){
		cout<<"\n====================\n Iter "<<h<<"\t"<<ub<<"\n =========================== \n ";
		 //==========================
		// Resolve e guarda a solução do problema relaxado
		//==========================
		Relaxed_Value = Solve_relaxed_model();
		Cria_problema_reduzido(x_bar);

/*		if(cplex.getStatus() == IloAlgorithm::Optimal){
			GetSolution();
			Display_configuration();
			Relaxed_Value = cplex.getObjValue();
			cout<<"oie "<<endl;

		}
		else{
			cout<<"Relaxado inviável \n \n \n \n"<<endl;
			exit(0);
		}
	*/

		//============================
		//  Resolve o problema reduzido como mip
		//=========================

		bool OptimalMIP = Solve_mip_model();
		if(OptimalMIP){
			Display_mip_configuration();
			initial_solution = true;
			if(ub > cplex.getObjValue() - EPSILON){
				ub = cplex.getObjValue();
				x_best = x_til;
			}
		}
		//=========================
		// Monta e adiciona os pseudo cortes
		//=======================
		IloExpr branch(env);
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				if(ceil(x_bar[i][j]) - x_bar[i][j] <= EPSILON){
					branch += vx[i][j] * ( 1 - x_bar[i][j]) + x_bar[i][j] * (1 - vx[i][j]);
				}
			}
		}
		mod.add( branch >= 1);
    branch.end();
    Reinitialize_bounds();
    h++;
    if(h >= 3){
			criterioparada = true;
		}
	}
}

void Mono::Reinitialize_bounds(){
	//=============================
	// Retornando aos bounds iniciais
	//=============================

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
				vx[i][j].setBounds(0, 1);
		}
	}
}

void Mono::Cria_problema_reduzido(vector<vector<double>> x){
	//=============================
	// Criar o problema reduzido
	//=============================

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(x[i][j] <= EPSILON || x[i][j] >= 1- EPSILON){ // Se solução binária
				vx[i][j].setBounds(x[i][j], x[i][j]);
			}
		}
	}
}

void Mono::Display_best_configuration(){
	/*printf("\n Caixas:");
	for(int j = 0; j < n; j++){
		if( y_best[j] >= EPSILON){
			printf(" %d\t", j+1);
		}
	}
	cout<<endl;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if( x_best[i][j] >= EPSILON){
				printf(" %d (%d) \t",i +1,j+1);
			}
		}
	}
	cout<<endl;*/
}
