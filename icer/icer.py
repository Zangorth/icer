###########
# Imports #
###########
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sea
import pandas as pd
import numpy as np

########
# ICER #
########
class IcePlot:
    def __init__(self, estimator, x, feature, feature_type='numeric'):
        '''
        Description - Generate an ICE Plot for a provided feature; identify clusters of observations within that
                      plot which behave similarly; and describe the characteristics of those clusters

        Parameters
        ----------
        estimator : callable
            A fitted model with a predict or predict proba method
        x : pd.DataFrame
            A dataframe of observations to be used in fitting the ice plot
        feature : str
            A string indicating the feature to be plotted and explained; must be a column in the x dataframe
        feature_type : str
            A string indicating whether the feature is categoric or numeric; used for determining plot type
            Options: 'categoric' or 'numeric'
        '''
        self.estimator = estimator
        self.x = x
        self.feature = feature
        self.feature_type = feature_type

        self.freezer = pd.DataFrame(columns=['index', 'feature', 'value', 'prediction'])
        self.pdp = None
        self.slopes = None
        self.clusters = None
        self.cluster_features = None

    def ice_data(self, return_data=True, outlier_removal=True):
        '''
        Description - Generate predictions for different values of the feature to be used in the ICE plot

        Parameters
        ----------
        return_data : boolean
            Indicate whether you would like the method to return the generated dataset of values and predictions
        outlier_removal : boolean
            Indicate whether extreme values on the feature should be removed during data generation
        '''
        ice_frame = self.x.copy()

        if outlier_removal and self.feature_type == 'numeric':
            ice_frame = ice_frame.loc[(ice_frame[self.feature] >= ice_frame[self.feature].quantile(0.025)) &
                                      (ice_frame[self.feature] <= ice_frame[self.feature].quantile(0.975))]

        bins = self.binning(ice_frame)

        # Parallelize this as necessary
        for value in bins:
            ice_frame[self.feature] = value
            append_frame = pd.DataFrame({'index': ice_frame.index.tolist(),
                                         'feature': self.feature,
                                         'value': value,
                                         'prediction': self.estimator.predict_proba(ice_frame)[:, 1]})

            self.freezer = pd.concat([self.freezer, append_frame], axis=0)

        self.pdp = self.freezer.groupby('value')['prediction'].mean().rename('prediction').reset_index()

        if return_data:
            return self.freezer


    def ice_plot(self, subsample=100, return_plot=False):
        '''
        Description - Generates the ICE plot
                      If slope_generator has not been run it will be run with a 2nd degree polynomial fit
                      If cluster_generator has been run lines will be colored based on the cluster they belong to

        subsample : int
            Number of lines to include on the ICE plot
        return_plot : boolean
            Indicator for whether to return the fig, ax objects for further customization
        '''
        if self.pdp is None:
            self.ice_data(return_data=False)

        sampled = np.random.choice(self.freezer['index'].unique(), size=subsample, replace=False)

        fig, ax = plt.subplots(figsize=(16, 9))
        for i in range(subsample):
            subset = self.freezer.loc[self.freezer['index'] == sampled[i]]

            if self.slopes is None:
                self.slope_generator()

            if self.clusters is not None:
                color_num = self.clusters.loc[self.clusters['index_tuple'] == sampled[i], 'cluster'].item()
                plot_kwargs = {'color': sea.color_palette('gist_rainbow',
                                                          self.clusters['cluster'].nunique()).as_hex()[color_num],
                               'alpha': 0.25}

            else:
                color_num = self.slopes.loc[self.slopes.index == sampled[i], 'ranking'].item()
                plot_kwargs = {'color': sea.color_palette('binary', 100).as_hex()[color_num],
                               'alpha': 0.25}

            # Update this to vary based on slope color;
            #   use kwargs to allow it to vary depending on if slopes have been defined
            ax.plot(subset['value'], subset['prediction'], **plot_kwargs)

        ax.plot(self.pdp['value'], self.pdp['prediction'], color='red', ls='dashed')

        if return_plot:
            return fig, ax

    def slope_generator(self, polynomial=2):
        '''
        Description - Estimates the slope on the predictions for each observation in the x dataframe

        Parameters
        ----------
        polynomial : int
            The degree of polynomial to be used in estimating the slope
            Default 2
        '''
        if self.pdp is None:
            self.ice_data(return_data=False)

        coef_names = [f'x**{i}' for i in np.arange(polynomial, 0, -1)] + ['Intercept', 'Total Movement']
        slopes = pd.DataFrame(self.freezer.groupby('index').apply(coef_getter, polynomial), columns=['coeffs'])
        slopes = pd.DataFrame(slopes['coeffs'].tolist(), columns=coef_names, index=slopes.index)

        slopes['ranking'] = MinMaxScaler((0.2, 0.8)).fit_transform(slopes['Total Movement'].values.reshape(-1, 1))
        slopes['ranking'] = (slopes['ranking']*100).round().astype(int)

        # Add in checks to only do this if it actually is a multi-index, idk what will happen if it's not
        slopes.index = pd.MultiIndex.from_tuples(slopes.index)
        slopes.index.names = self.x.index.names

        self.slopes = slopes.copy()

        return None

    def cluster_generator(self, n_clusters=2, feature_reduction=None):
        '''
        Description - Separates individuals out into kmeans clusters based on their slopes and other features included
                      in the x dataframe

        Parameters
        ----------
        n_clusters : int
            Number of clusters to generate
            Default 2
        feature_reduction : int
            Number of x features to use in clustering
            Default None, which will include all features
        '''
        # Add feature selection for the clusters
            # Forward Feature Selection, calculate the PDP of each cluster and maximize distance between the PDP lines
        if self.slopes is None:
            self.slope_generator()

        cluster_frame = self.slopes[[col for col in self.slopes.columns if 'x**' in col]].copy()
        cluster_frame = cluster_frame.merge(self.x, left_index=True, right_index=True)
        cluster_frame = pd.DataFrame(MinMaxScaler().fit_transform(cluster_frame),
                                     columns=cluster_frame.columns, index=cluster_frame.index)

        self.cluster_features = cluster_frame.columns
        if feature_reduction is not None:
            pass

        kmeans = KMeans(n_clusters=n_clusters, random_state=52)
        kmeans.fit(cluster_frame[self.cluster_features])

        cluster_frame['cluster'] = kmeans.labels_
        cluster_frame['index_tuple'] = cluster_frame.index.tolist()

        self.clusters = cluster_frame.copy()

        return None

    def cluster_analysis(self, cluster_number=0, return_plot=False):
        '''
        Description - Provides violin plots showing the differences between a cluster and all other observations

        Parameters
        ----------
        cluster_number : int
            Which cluster to analyze; clusters start at 0 and end at the n_clusters specified in cluster_generator
        return_plot : boolean
            Indicator for whether to return the fig, ax objects for further customization
        '''
        if self.clusters is None:
            self.cluster_generator()

        cluster_frame = self.clusters.copy()

        palette = sea.color_palette('gist_rainbow', n_colors=cluster_frame['cluster'].nunique()+1)
        palette = [palette[cluster_number], palette[-1]]

        cluster_frame['coi'] = np.where(cluster_frame['cluster'] == cluster_number,
                                        f'Cluster {cluster_number}', 'Everyone Else')

        cluster_frame = cluster_frame.sort_values('coi')

        cluster_frame = cluster_frame.drop([col for col in cluster_frame.columns if '**' in col] +
                                           ['index_tuple', 'cluster'],
                                           axis=1)

        importance = cluster_frame.groupby('coi').mean().diff().mean().abs().sort_values(ascending=False)[0:5]

        fig, axes = plt.subplots(ncols=len(importance), sharex=True, figsize=(18, 9))
        for i in range(len(importance)):
            sea.violinplot(y=importance.index[i], data=cluster_frame, palette=palette, hue='coi',
                           split=True, common_norm=True, ax=axes[i])

            if i == 0:
                axes[i].legend(loc='best')

            else:
                axes[i].legend().set_visible(False)

            axes[i].set_yticks([])

        fig.suptitle(f'Analysis of Cluster {cluster_number}')

        if return_plot:
            return fig, axes

    def binning(self, ice_frame):
        optimal_bin_count = int(np.ceil(np.log2(len(ice_frame[self.feature]))) + 1)
        bin_count = min([optimal_bin_count, ice_frame[self.feature].nunique(), 100])

        if self.feature_type == 'numeric' and ice_frame[self.feature].nunique() == bin_count:
            bins = sorted(ice_frame[self.feature].unique())

        elif self.feature_type == 'numeric':
            bins = np.histogram(ice_frame.loc[ice_frame[self.feature].notnull(), self.feature], bins=bin_count)
            bins = sorted(bins[1])

        else:
            bins = ice_frame[self.feature].value_counts().sort_values(ascending=False).reset_index()[0:bin_count]

        return bins


######################
# Group Coefficients #
######################
def coef_getter(df, n=1):
    model = np.poly1d(np.polyfit(df['value'].astype(float), df['prediction'].astype(float), n))
    total_movement = pd.Series(model(df['value'])).diff().abs().sum()

    try:
        return [model.coeffs[i] for i in range(n+1)] + [total_movement]
    except IndexError:
        return [0 for i in range(n+2)]
