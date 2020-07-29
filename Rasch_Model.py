#!/usr/bin/env python
# coding: utf-8

import pystan
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr



model = '''
data {
  int<lower=1> J;              // number of students
  int<lower=1> K;              // number of questions
  int<lower=1> N;              // number of observations
  int<lower=1,upper=J> jj[N];  // student for observation n
  int<lower=1,upper=K> kk[N];  // question for observation n
  int<lower=0,upper=1> y[N];   // correctness for observation n
}

parameters {
  real delta;         // mean student ability
  real alpha[J];      // ability of student j - mean ability
  real beta[K];       // difficulty of question k
}

model {
  alpha ~ std_normal();         // informative true prior
  beta ~ std_normal();          // informative true prior
  delta ~ normal(0.75, 1);      // informative true prior
  for (n in 1:N)
    y[n] ~ bernoulli_logit(alpha[jj[n]] - beta[kk[n]] + delta);
}
'''


n_s = 400 # number of students
n_i = 100 # number of items

# pick a random ability for each student from an N(0,1) distribution

abilities = np.random.randn(n_s,1)

# pick a random difficulty for each item from an N(0,1) distribution

difficulties = np.random.randn(1,n_i)

# the IRT model says that P(correct[s,i]) = logistic(ability[s] -difficulty[i])

def logistic(a):
    return 1./(1+np.exp(-a))
prob_correct = logistic(abilities - difficulties)

print (abilities.shape, difficulties.shape, prob_correct.shape)

# flip a coin to pick 'correct' or 'incorrect' for each student based on the
# probability of a correct response

student_responses = np.random.binomial(1,prob_correct)
#print (student_responses)


J = student_responses.shape[0]
K = student_responses.shape[1]
N = student_responses.shape[0] * student_responses.shape[1]
jj = [0] * N
kk = [0] * N
y = [0] * N

for i in range(N):
    jj[i] = i % 400
    jj[i] += 1

for i in range(N):
    kk[i] = i % 100
    kk[i] += 1

for i in range(N):
    y[i] = student_responses[int(i/100)][i%100]

numpy_array = np.array(jj)
jj = numpy_array.T

numpy_array = np.array(kk)
kk = numpy_array.T


dat = {'J': J,
       'K': K,
       'N': N,
       'jj': jj,
       'kk': kk,
       'y': y}

sm = pystan.StanModel(model_code=model)
#fit = sm.sampling(data=dat, iter=100, chains=1)
fit = sm.sampling(data=dat)

#rint(fit)
la = fit.extract(permuted=True)  # return a dictionary of arrays
delta = la['delta']
alpha = la['alpha']
beta = la['beta']
## return an array of three dimensions: iterations, chains, parameters
##a = fit.extract(permuted=False)

print('delta : ',delta)
print('delta size : ',len(delta))
print('alpha : ',alpha)
print('alpha size : ',len(alpha))
print('alpha individual size : ',len(alpha[0]))
print('beta : ',beta)
print('beta size : ',len(beta))
print('beta individual size : ',len(beta[0]))

flattened = []
for sublist in abilities:
    for val in sublist:
        flattened.append(val)


corr, _ = pearsonr(alpha[0]+delta[0], flattened)
print('Pearsons correlation: %.3f' % corr)

corr, _ = spearmanr(alpha[0] + delta[0], flattened)
print('Spearmans correlation: %.3f' % corr)
