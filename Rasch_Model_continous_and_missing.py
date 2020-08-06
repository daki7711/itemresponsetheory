#!/usr/bin/env python
# coding: utf-8

import pystan
from scipy.special import expit
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def logistic(a):
    return expit(a)

# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)

model = '''
data {
  int<lower=1> J;              // number of students
  int<lower=1> K;              // number of questions
  real<lower=0> kappa;           // kappa for beta distribution
  int<lower=1> N_obs;              // number of observations
  int<lower=1,upper=J> jj_obs[N_obs];  // student for observation n
  int<lower=1,upper=K> kk_obs[N_obs];  // question for observation n
  real<lower=0,upper=1> y_obs[N_obs];   // correctness for observation n

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

  for (n in 1:N_obs){
    real in_logit;
    in_logit = alpha[jj_obs[n]] - beta[kk_obs[n]] + delta;
    if(in_logit <= 0) in_logit=0.000001;
    if(in_logit >= 1) in_logit=0.999999;
    real mean = inv_logit(in_logit);

    y_obs[n] ~ beta_proportion(mean, kappa);
  }
}
'''


spearman_ability = []
spearman_item = []
pearson_ability = []
pearson_item = []

print("Starting the iteration")
for i in range(1,10):
    spearman_ability_temp = []
    spearman_item_temp = []
    pearson_ability_temp = []
    pearson_item_temp = []
    for j in range(5):
        n_s = 100  # number of students
        n_i = 100  # number of items
        # pick a random ability for each student from an N(0,1) distribution
        abilities = np.random.randn(n_s, 1)
        # pick a random difficulty for each item from an N(0,1) distribution
        difficulties = np.random.randn(1, n_i)
        # the IRT model says that P(correct[s,i]) = logistic(ability[s] -difficulty[i])
        prob_correct = logistic(abilities - difficulties)
        # flip a coin to pick 'correct' or 'incorrect' for each student based on the
        # probability of a correct response

        # for continuous variables added a epsilon value on the probabilty of correction
        w = 10.0
        a = np.array(w * prob_correct)
        b = np.array(w * (1 - prob_correct))
        student_responses = np.random.beta(a, b)

        for k in range(n_s):
            for l in range(n_i):
                value = np.random.binomial(1, (10 - i) * 0.1)
                if value == 0:
                    student_responses[k][l] = -1

        # This is the part where the student ability and the item difficulty is being trained
        J = student_responses.shape[0]
        K = student_responses.shape[1]

        N_obs = 0
        N_miss = 0
        for k in range(n_s):
            for l in range(n_i):
                if student_responses[k][l] != -1:
                    N_obs += 1
                else:
                    N_miss += 1

        jj_obs = [0] * N_obs
        kk_obs = [0] * N_obs
        y_obs = [0.0] * N_obs

        jj_miss = [0] * N_miss
        kk_miss = [0] * N_miss


        observe = 0
        missing = 0
        for k in range(n_s):
            for l in range(n_i):
                if student_responses[k][l] != -1:
                    jj_obs[observe] = k + 1
                    kk_obs[observe] = l + 1
                    y_obs[observe] = student_responses[k][l]
                    observe += 1
                else:
                    jj_miss[missing] = k + 1
                    kk_miss[missing] = l + 1
                    missing += 1

        dat = {'J': J,
               'K' : K,
               'kappa' : w,
               'N_obs' : N_obs,
               'jj_obs' : jj_obs,
               'kk_obs' : kk_obs,
               'y_obs'  : y_obs
               }

        sm = pystan.StanModel(model_code=model)
        fit = sm.sampling(data=dat)
        #fit = sm.sampling(data=dat,init_r=0.1)

        la = fit.extract(permuted=True)  # return a dictionary of arrays
        delta = la['delta']
        alpha = la['alpha']
        beta = la['beta']

        flattened_abilities = []
        for sublist in abilities:
            for val in sublist:
                flattened_abilities.append(val)

        flattened_difficulty = []
        for sublist in difficulties:
            for val in sublist:
                flattened_difficulty.append(val)

        alpha = alpha[int(alpha.shape[0] / 2):]
        alpha = np.mean(alpha, axis=0)
        delta = np.mean(delta, axis=0)

        beta = beta[int(beta.shape[0] / 2):]
        beta = np.mean(beta, axis=0)

        corr, _ = pearsonr(alpha + delta, flattened_abilities)
        pearson_ability_temp.append(corr)

        corr, _ = spearmanr(alpha + delta, flattened_abilities)
        spearman_ability_temp.append(corr)

        corr, _ = pearsonr(beta, flattened_difficulty)
        pearson_item_temp.append(corr)

        corr, _ = spearmanr(beta, flattened_difficulty)
        spearman_item_temp.append(corr)

        print("Finished ", j, "th iteration of ", i * 10, " percent missing data")
    spearman_ability.append(Average(spearman_ability_temp))
    spearman_item.append(Average(spearman_item_temp))
    pearson_ability.append(Average(pearson_ability_temp))
    pearson_item.append(Average(pearson_item_temp))


plt.title('Continuous IRT Resilience Test using STAN')
plt.xlabel('Percent of Missing Data')
plt.ylabel('Correlation')
plt.plot([10,20,30,40,50,60,70,80,90],spearman_ability,'go', label = 'spearman_ablilty')
plt.plot([10,20,30,40,50,60,70,80,90],pearson_ability,'ro', label = 'pearson ability')
plt.plot([10,20,30,40,50,60,70,80,90],spearman_item,'g^', label = 'spearman item')
plt.plot([10,20,30,40,50,60,70,80,90],pearson_item,'r^', label = 'pearson item')
plt.legend(loc="upper right")
plt.savefig('Correlation graph(beta distribution).png')
