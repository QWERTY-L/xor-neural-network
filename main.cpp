#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

void printList(std::vector<std::vector<double>> lst) {
	std::cout << "\n";
	for (int i = 0; i < lst.size(); i++) {
		std::cout << "\n";
		for (int j = 0; j < lst[i].size(); j++) {
			std::cout << lst[i][j] << ", ";
		}
	}
	std::cout << "\n";
}

double relu(double x) {

	if (x > 0) {

		return x;

	}
	else {

		return 0;

	}

}

std::vector<double> relu(std::vector<double> x) {
	std::vector<double> c(x.size());
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = relu(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> relu(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

			c[i] = relu(x[i]);

	}
	return c;

}

/*double crossEntropy(double yp, double y) {

	if(y == 1){

		return -log(yp);

	} else{

		return -log(1-yp);

	}

}

std::vector<double> crossEntropy(std::vector<double> yp, std::vector<double> y) {
	std::vector<double> c(y.size());
	#pragma omp parallel for
	for (int i = 0; i < y.size(); i++) {

		c[i] = crossEntropy(yp[i], y[i]);

	}
	return c;

}

std::vector<std::vector<double>> crossEntropy(std::vector<std::vector<double>> yp, std::vector<std::vector<double>> y) {
	std::vector<std::vector<double>> c(y.size(), std::vector<double>(1));
	#pragma omp parallel for
	for (int i = 0; i < y.size(); i++) {

			c[i] = crossEntropy(yp[i], y[i]);

	}
	return c;

}*/

double drelu(double x) {

	if (x > 0) {

		return 1;

	}
	else {

		return 0;

	}

}

std::vector<double> drelu(std::vector<double> x) {
	std::vector<double> c(x.size());
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = drelu(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> drelu(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = drelu(x[i]);

	}
	return c;

}

double sig(double x) {

	return 1.0 / (1 + exp(-x));

}

std::vector<double> sig(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = sig(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> sig(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = sig(x[i]);

	}
	return c;

}

double dsig(double a) { //takes input sig(a) and returns dsig(a)

	//double a = sig(x);
	return (a * (1 - a));

}

std::vector<double> dsig(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dsig(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> dsig(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dsig(x[i]);

	}
	return c;

}


/*double tanh(double x) {

	return 1.0 / (1 + exp(-x));

}*/ //this is a builtin function of math.h

std::vector<double> tanh(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = tanh(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> tanh(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = tanh(x[i]);

	}
	return c;

}

double dtanh(double a) { //takes input tanh(a) and returns dtanh(a)

	//double a = sig(x);
	return (1 - (a*a));

}

std::vector<double> dtanh(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dtanh(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> dtanh(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dtanh(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> a) {
	
	std::vector<std::vector<double>> c(a[0].size(), std::vector<double>(a.size()));
	#pragma omp parallel for
	for (int i = 0; i < a[0].size(); i++) {
	#pragma omp parallel for
		for (int j = 0; j < a.size(); j++) {

			c[i][j] = a[j][i];
		}

	}
	return c;
}

std::vector<double> add(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] + b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> add(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

			c[i] = add(a[i], b[i]);

	}
	return c;

}

std::vector<double> sum(std::vector<double> a) {

	std::vector<double> c = { 0 };
	for (int i = 0; i < a.size(); i++) {

		c[0] += a[i];

	}
	return c;

}

std::vector<std::vector<double>> sum(std::vector<std::vector<double>> a) {

	std::vector<std::vector<double>> c(a.size(), std::vector < double>(1));
	for (int i = 0; i < a.size(); i++) {

		c[i] = sum(a[i]);

	}
	return c;

}

std::vector<double> sub(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] - b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> sub(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = sub(a[i], b[i]);

	}
	return c;

}

std::vector<double> emult(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] * b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> emult(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = emult(a[i], b[i]);

	}
	return c;

}

std::vector<double> smult(double a, std::vector<double> b) {

	std::vector<double> c(b.size());
	#pragma omp parallel for
	for (int i = 0; i < b.size(); i++) {

		c[i] = a * b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> smult(double a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(b.size(), std::vector<double>(b[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < b.size(); i++) {

		c[i] = smult(a, b[i]);

	}
	return c;

}

std::vector<std::vector<double>> dadd(std::vector<std::vector<double>> a, std::vector<double> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

			c[i] = add(a[i], b);

	}
	return c;

}

void printList(std::vector<double> lst) {

	std::cout << "\n";
	for (int i = 0; i < lst.size(); i++) {

		std::cout << lst[i] << ", ";

	}
	std::cout << "\n";
}

double dot(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	double out = 0;
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] * b[i];
	}
	for (int i = 0; i < a.size(); i++) {

		out += c[i];

	}
	return out;

}
std::vector<double> dot(std::vector<double> a, std::vector<std::vector<double>> b) {

	std::vector<double> c(b.size());
	#pragma omp parallel for
	for (int i = 0; i < b.size(); i++) {

		c[i] = dot(a, b[i]);

	}
	return c;

}

std::vector<std::vector<double>> dot(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = dot(a[i], b);

	}
	return c;
}

std::vector<std::vector<double>> h(std::vector<std::vector<double>> x, std::vector<std::vector<double>> w, std::vector<double> b, std::vector<std::vector<double>>(*act)(std::vector<std::vector<double>>)) {

	return act(dadd(transpose(dot(w, x)), b));

}

double MSE(double yh, double yp) {

	double c = (yh + yp);
	return (c * c);

}

std::vector<double> MSE(std::vector<double> yh, std::vector<double> yp) {

	std::vector<double> c = sub(yh, yp);
	return emult(c, c);

}

std::vector<std::vector<double>> MSE(std::vector<std::vector<double>> yh, std::vector<std::vector<double>> yp) {

	std::vector<std::vector<double>> c = sub(yh, yp);
	return emult(c, c);

}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>, std::vector<double>, std::vector<std::vector<double>>> GradientDescent(int samples, double alpha, std::vector<std::vector<double>> X, std::vector<std::vector<double>> y, std::vector<std::vector<double>> w1, std::vector<double> b1, std::vector<std::vector<double>> w2, std::vector<double> b2, std::vector<std::vector<double>>(*act1)(std::vector<std::vector<double>>), std::vector<std::vector<double>>(*act2)(std::vector<std::vector<double>>), std::vector<std::vector<double>> dw1, std::vector<double> db1, std::vector<std::vector<double>> dw2, std::vector<double> db2, std::vector<std::vector<double>>(*dact1)(std::vector<std::vector<double>>), std::vector<std::vector<double>>(*dact2)(std::vector<std::vector<double>>)) {

	//returns w1, w2, b1, b2 so they can be stored in main()

	//evaluation
	std::vector<std::vector<double>> L1 = h(X, w1, b1, act1);
	std::vector<std::vector<double>> yp = h(L1, w2, b2, act2);

	//DERIVATIVES

	//Layer 2
	std::vector<std::vector<double>> dMSE = smult(2, sub(transpose(y), yp)); //by the power rule, dx^2/dx = 2x (the 2 can technically be removed to give the effect of a twice as small learning rate. This may negligibly improve the performance)
	std::vector<std::vector<double>> dACT2 = dact2(yp);
	std::vector<std::vector<double>> dACT1 = dact1(L1);
	std::vector<std::vector<double>> dz2 = emult(dMSE, dACT2);
	dw2 = dot(transpose(dz2), transpose(L1)); //technically, dw2 is the negative of this, but we omit the negative and later add rather than subtract
	db2 = dot(transpose(dMSE), transpose(dACT2))[0]; //this is also the negative of db2
	
	//Layer 1
	std::vector<std::vector<double>> dz1 = emult(dot(dz2, transpose(w2)), dACT1);
	dw1 = transpose(dot(transpose(dz1), transpose(X))); //again, these are the negatives of dw1 and db1 respectivly
	db1 = transpose(sum(transpose(dz1)))[0];

	//update weights and biases

	w1 = add(w1, (smult(alpha, smult(1.0 / samples, transpose(dw1))))); //note, remove 1 smult and change to smult(alpha/samples, vector);
	w2 = add(w2, (smult(alpha, smult(1.0 / samples, dw2))));
	b1 = add(b1, (smult(alpha, smult(1.0 / samples, db1))));
	b2 = add(b2, (smult(alpha, smult(1.0 / samples, db2))));

	/*
	printList(w1);
	printList(b1);
	printList(w2);
	printList(b2);
	*/
	//printList(smult(1.0 / samples, sum(MSE(y, yp))));
	//printList(yp);
	std::vector<std::vector<double>> losss = smult(1.0 / samples, sum(MSE(y, yp)));

	return std::make_tuple(w1, w2, b1, b2, losss);

}

int main() {

	std::cout.precision(17);
	srand((unsigned int)time(NULL));
	std::vector<std::vector<double>> X = { {0, 0}, {1, 1}, {1, 0}, {0, 1} };
	std::vector<std::vector<double>> y = { {0, 0, 1, 1} };
	std::cout << "X: ";
	printList(X);
	std::cout << "\n\n^\nY:";
	printList(y);
	std::vector<std::vector<double>>(*act1)(std::vector<std::vector<double>>);
	std::vector<std::vector<double>>(*act2)(std::vector<std::vector<double>>);
	act1 = tanh;
	act2 = sig;

	/*for(int i=0;i<1000;i++){

		std::cout << 2 * ((float) rand()/RAND_MAX) - 1 << std::endl;

	}*/
	//std::cout << RAND_MAX;

	for(int i=0;i<rand();i++){ //make random numbers more random

		//for randomness
		sig(42);
	}


	//initialise weights and biases
	std::vector<std::vector<double>> w1(2, std::vector<double>(2, 0)); //format (L_i, (L_i-1, 0)) where L_i is the wieghts in the current layer and L_i-1 is from the previous layer
	std::vector<double> b1(2, 0); //format (L_i, 0)
	std::vector<std::vector<double>> w2(1, std::vector<double>(2, 0)); //format (L_i, (L_i-1, 0))
	std::vector<double> b2(1, 0); //format (L_i, 0)

	//std::vector<std::vector<double>> w1 = { {2 * ((float) rand()/RAND_MAX) - 1, 2 * ((float) rand()/RAND_MAX) - 1, 2 * ((float) rand()/RAND_MAX) - 1 }, {2 * ((float) rand()/RAND_MAX) - 1, 2 * ((float) rand()/RAND_MAX) - 1, 2 * ((float) rand()/RAND_MAX) - 1} };
	for(int i=0;i<w1.size();i++){

		for(int j=0;j<w1[0].size();j++){

			w1[i][j] = 2 * ((float) rand()/RAND_MAX) - 1;

		}

	}

	for(int j=0;j<b1.size();j++){

		b1[j] = 2 * ((float) rand()/RAND_MAX) - 1;

	}

	for(int i=0;i<w2.size();i++){

		for(int j=0;j<w2[0].size();j++){

			w2[i][j] = 2 * ((float) rand()/RAND_MAX) - 1;

		}

	}

	for(int j=0;j<b2.size();j++){

		b2[j] = 2 * ((float) rand()/RAND_MAX) - 1;

	}

	/*
	printList(transpose(dot(w1, X)));
	std::cout << std::endl;
	printList(b1);
	std::vector<std::vector<double>> a = h(X, w1, b1, relu);
	printList(a);
	std::vector<std::vector<double>> b = h(a, w2, b2, relu);
	printList(b);
	//std::vector<double> n = add(b1, b1);
	//printList(dot(X, X));
	//printList(n);
	//std::cout << omp_get_thread_num();
	std::vector<std::vector<double>> c = MSE(transpose(y), b);
	printList(c);
	printList(smult(0.25, sum(transpose(c))));
	*/

	//set gd variables
	double alpha = 0.5;
	int batches = 100000; //if under 10,000 edit gradient descent for loop (i < batches/10000)
	std::vector<std::vector<double>> loss;

	//intialize derivatives
	std::vector<std::vector<double>> dw1 = w1;
	std::vector<double> db1 = b1;
	std::vector<std::vector<double>> dw2 = w2;
	std::vector<double> db2 = b2;

	//gradient descent
	for (int i = 0; i < batches/10000; i++) {
	
		for(int j = 0; j<10000; j++){	
			auto r_vals = GradientDescent(X.size(), alpha, X, y, w1, b1, w2, b2, act1 /*change to relu after*/, act2, dw1, db1, dw2, db2, dtanh /*drelu*/, dsig);
			w1 = std::get<0>(r_vals);
			w2 = std::get<1>(r_vals);
			b1 = std::get<2>(r_vals);
			b2 = std::get<3>(r_vals);
			loss = std::get<4>(r_vals);
		}
	std::cout << "Loss: ";
	printList(loss);
	if(i==5){//check if NN is converging

		std::vector<std::vector<double>> yp = (h(h(X, w1, b1, act1), w2, b2, act2));//yp[i][0]
		//printList(yp);
		//{ {0, 0, 1, 1} }
		if(yp[0][0] > 0.3 || yp[1][0] > 0.3 || yp[2][0] < 0.7 || yp[3][0] < 0.7){

			std::cout << "Neural Network not converging.\n\nRe-initialising weights";
			for(int i=0;i<w1.size();i++){

				for(int j=0;j<w1[0].size();j++){

					w1[i][j] = 2 * ((float) rand()/RAND_MAX) - 1;

				}

			}

			for(int j=0;j<b1.size();j++){

				b1[j] = 2 * ((float) rand()/RAND_MAX) - 1;

			}

			for(int i=0;i<w2.size();i++){

				for(int j=0;j<w2[0].size();j++){

					w2[i][j] = 2 * ((float) rand()/RAND_MAX) - 1;

				}

			}

			for(int j=0;j<b2.size();j++){

				b2[j] = 2 * ((float) rand()/RAND_MAX) - 1;

			}

			i=0;
			}

	}

	}

	//Input - Output System
	std::string cmnd = "";
	while (true) {

		std::cout << std::endl;
		std::cout << ">>> ";
		std::getline(std::cin, cmnd);
		//cout << cmnd;
		int split = cmnd.find(" ");
		std::string s1 = cmnd.substr(0, split);
		//cout << split;
		std::string r = cmnd.substr(split + 1);
		int split2 = r.find(" ");
		std::string r1 = r.substr(0, split2);
		std::string r2 = r.substr(split2 + 1);
		if (s1 == "end") {

			break;

		}

		if (s1 == "btrain") {

			batches = stoi(r);
			for (int i = 0; i < batches/10000; i++) {
			
				for(int j = 0; j<10000; j++){	
					auto r_vals = GradientDescent(X.size(), alpha, X, y, w1, b1, w2, b2, tanh /*change to relu after*/, sig, dw1, db1, dw2, db2, dtanh /*drelu*/, dsig);
					w1 = std::get<0>(r_vals);
					w2 = std::get<1>(r_vals);
					b1 = std::get<2>(r_vals);
					b2 = std::get<3>(r_vals);
					loss = std::get<4>(r_vals);
				}
			std::cout << "Loss: ";
			printList(loss);

		}
		
		
		}

		if (s1 == "eval") {

			printList(h(h({ { std::stod(r1), std::stod(r2) } }, w1, b1, act1), w2, b2, act2));

		}

		/*if (s1 == "edit") {

			if (r[0] == '-' && r[1] == 'i') {

				//edit single index: todo
				cout << "you typed 'edit -i'";

			}
			//int rplit = r.find(" ");
			//std::string r1 = r.substr(0, rplit);
			//std::string r2 = r.substr(rplit);
			if (r == "x") {

				double bar;
				for (int foo = 0; foo < samples; foo++) {

					cout << endl << "Enter Number " << (foo + 1) << endl << ">>> ";
					cin >> bar;
					x[foo] = bar;

				}
				printList(x, samples);

			}

			if (r == "y") {

				double bar;
				for (int foo = 0; foo < samples; foo++) {

					cout << endl << "Enter Number " << (foo + 1) << endl << ">>> ";
					cin >> bar;
					y[foo] = bar;

				}
				printList(y, samples);

			}

			if (r == "samples") {

				cout << endl << "Enter number of samples: " << endl << ">>> ";
				cin >> samples;
				cout << endl << "There are now " << samples << " samples." << endl << "WARNING, MAKE SURE TO CHANGE 'x' AND 'y' USING 'edit x' and 'edit y'";

			}

		}*/

		if (s1 == "print") {
			std::cout.precision(10);

			if (r == "x") {

				std::cout << "x:";
				printList(X);

			}

			if (r == "y") {

				std::cout << "y:";
				printList(y);

			}

			if (r == "w1") {

				std::cout << "w1:";
				printList(w1);

			}
			if (r == "b1") {

				std::cout << "b1:";
				printList(b1);

			}
			if (r == "w2"){
				std::cout << "w2:";
				printList(w2);

			}
			if (r == "b2") {

				std::cout << "b2:";
				printList(b2);

			}
			if (r == "samples") {

				std::cout << "samples:" << std::endl << X.size();

			}

		}

	}


	return 0;

}