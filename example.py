###########
# Imports #
###########
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sea
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

#############
# Load Data #
#############
for dataset in sea.get_dataset_names():
    print(f'{dataset}: {len(sea.load_dataset(dataset))}')

# Diamonds large alternative
df = sea.load_dataset('iris')

y = np.where(df['species'] == 'setosa', 1, 0)
x = df.drop('species', axis=1)

x['petal_width_category'] = 1
x.loc[x['petal_width'] < x['petal_width'].quantile(0.33), 'petal_width_category'] = 0
x.loc[x['petal_width'] > x['petal_width'].quantile(0.66), 'petal_width_category'] = 2

x = x.drop('petal_width', axis=1)
x['Index1'] = 'Some'
x['Index2'] = 'Bullshit'
x['Index3'] = np.arange(len(x))
x = x.set_index(['Index1', 'Index2', 'Index3'])

estimator = RandomForestClassifier()
estimator.fit(x, y)


ice = IcePlot(estimator, x, feature='sepal_length')
ice.ice_data()
ice.ice_plot()

ice.cluster_generator()

ice.ice_plot()
ice.cluster_analysis(1)

ice = IcePlot(estimator, x, feature='petal_width_category', feature_type='numeric')
ice.ice_data('petal_width_category')
ice.ice_plot()


model = np.poly1d(np.polyfit(ice.pdp['value'], ice.pdp['prediction'], 2))
myline = np.linspace(4, 8, 100)

fig, ax = plt.subplots(figsize=(16, 9))
ax.scatter(ice.pdp['value'], ice.pdp['prediction'])
ax.plot(myline, model(myline))


for i in ice.freezer['index'].unique():
    print(i)
    subset = ice.freezer.loc[ice.freezer['index'] == i]
    coef_getter(subset)