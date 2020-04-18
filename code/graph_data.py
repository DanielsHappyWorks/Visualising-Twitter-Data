import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../output/model_data/neural_net_classifier/full_df.pkl")
print(df.head())

df.groupby(['Vader Pred','Classifier Pred']).size().unstack().plot(kind='bar',stacked=True)
plt.show()