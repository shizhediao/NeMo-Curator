### This script is borrowed and adapted from the RegMix project.
### https://github.com/sail-sg/regmix/blob/main/regression_fitting/regression.ipynb

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import lightgbm as lgb
from scipy.stats import spearmanr

# set objective
selected = 3
objective = "valid-avg"

csv_file_path = "lm_harness_results.csv"
df = pd.read_csv(csv_file_path)

shuffled_df = df.sample(frac=1, random_state=42)  # random_state for reproducibility
train_df = shuffled_df.iloc[:int(len(df)*0.9)]
test_df = shuffled_df.iloc[int(len(df)*0.9):]

# Print basic information
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")

# Define column names explicitly
config_columns = [
    'train_the_cluster1', 'train_the_cluster2', 'train_the_cluster3', 'train_the_cluster4',
    'train_the_cluster5', 'train_the_cluster6', 'train_the_cluster7', 'train_the_cluster8',
    'train_the_cluster9', 'train_the_cluster10', 'train_the_cluster11'
]

target_columns = [
    'piqa_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'valid_avg'
]
# 'wiki_ppl', 'lambda_ppl', 'lambda_acc', 'arc_challenge_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa', 'avg', 
    # 'mmlu_acc', 'mmlu_cloze_avg', 'mmlu_cloze_stem', 'mmlu_cloze_humanities', 'mmlu_cloze_social_sciences', 'mmlu_cloze_other'

train_df_config = train_df[config_columns]
train_df_target = train_df[target_columns]

test_df_config = test_df[config_columns]
test_df_target = test_df[target_columns]

print(train_df_config.head())
print(train_df_target.head())
print(test_df_config.head())
print(test_df_target.head())


X_train = train_df_config[train_df_config.columns[0:]].values
print(f"X_train.shape: {X_train.shape}")
y_train = train_df_target[train_df_target.columns[0:]].values
print(f"y_train.shape: {y_train.shape}")
X_test = test_df_config[test_df_config.columns[0:]].values
print(f"X_test.shape: {X_test.shape}")
y_test = test_df_target[test_df_target.columns[0:]].values
print(f"y_test.shape: {y_test.shape}")

KEY_METRICS = train_df_target.columns[0:18].tolist()
print(f"KEY_METRICS: {KEY_METRICS}")

df.isna().sum()

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    "num_iterations": 1000, 
    'seed': 42,
    'learning_rate': 1e-2,
    "verbosity": -1,
}

        
np.random.seed(42)

predictor = []

for i in range(len(KEY_METRICS)):

    target = y_train[:, i]
    test_target = y_test[:, i]
    
    gbm = lgb.LGBMRegressor(**hyper_params)

    reg = gbm.fit(X_train, target,
        eval_set=[(X_test, test_target)],
        eval_metric='l2', callbacks=[
        lgb.early_stopping(stopping_rounds=3, verbose=False),
    ])
    r, p = spearmanr(reg.predict(X_test), test_target)
    print(i, KEY_METRICS[i], "Correlation: {}".format(np.round(r*100, 2)))

    predictor.append(reg)
    # break

# Visualize the fitting on Pile-CC

data = {
    'True Loss': y_test[:, selected],
    'Pred Loss': predictor[selected].predict(X_test)
}

graph = sns.jointplot(data, x='Pred Loss', y='True Loss', kind='reg',
                      height=10, 
                      scatter_kws={'s': 128, 'color': '#5969CB'},
                  joint_kws={'line_kws': {
                      'color': '#C3364A', 
                                          'linewidth': 6,
                      'linestyle': 'dashed',
                  }},
                  marginal_kws={'line_kws': {
                      'color': '#5969CB', 
                      'linewidth': 6}}
             )

r, p = spearmanr(data['Pred Loss'], data['True Loss'])

phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)

graph.ax_joint.legend(
    [phantom],['Correlation: {:.2f}'.format(np.round(r, 2))],
    edgecolor='black', 
           fancybox=False,
           prop={'size': 32, }, 
           handlelength=-0.5
)

graph.ax_joint.set_ylabel('True Loss', fontdict={
    'size':48
})
graph.ax_joint.set_xlabel('Pred Loss', fontdict={
    'size':48
})

graph.ax_marg_x.remove()
graph.ax_marg_y.remove()

graph.ax_joint.grid(True, ls='dashed')
graph.ax_joint.spines[['right', 'top']].set_visible(True)

plt.savefig('lm_harness_results.pdf')


# simulation
np.random.seed(42)

# token distribution of each domain
prior_dist = [0.11328527, 0.07960865, 0.00391349, 0.1853759, 
              0.05108136, 0.01596293, 0.10175077, 0.00370752, 
              0.06652935, 0.00175077, 0.02708548]

samples = np.random.dirichlet(prior_dist * 1, 100000)
samples.shape

# The targe metric is set as the Pile-CC validation loss
simulation = predictor[selected].predict(samples)

plt.hist(simulation, bins=32)

plt.xlabel('Pred Loss')
plt.ylabel('Frequency')
print()

# take the average of top-k simulated data mixture as the optimal data mixture 
k = 128
top_k_samples = samples[np.argsort(simulation)[0:k]]
top_k_samples.shape

# you can get the optimal data mixture by taking the average of top-k samples
optimal_data_mixture = np.mean(top_k_samples, axis=0)
print(optimal_data_mixture)

df = pd.DataFrame(data=np.concatenate([np.array([prior_dist]), 
                                       top_k_samples], axis=0), columns=[i for i in train_df_config.columns[0:]])

df = pd.melt(df)
df['type'] = (['Human']+['RegMix']*top_k_samples.shape[0])*len(train_df_config.columns[0:])
df.head()
df.info()