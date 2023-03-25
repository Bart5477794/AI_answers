from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def visualize_data(data_dict, train):
    fig, ax = plt.subplots(1, figsize=(15, 15))
    for i in range(len(data_dict)):
        if train:
            x = np.array(data_dict[f'a{i + 1}']['strains']).squeeze()
        else:
            x = np.array(data_dict[f'te_a{i + 1}']['strains']).squeeze()
        ax.plot(x, label=f'specimens a{i + 1}')

    ax.set_xlabel('time steps')
    ax.set_ylabel('strains')
    ax.legend(loc='upper left')
    if train:
        fig.suptitle('Training raw data')
    else:
        fig.suptitle('Testing raw data')
    plt.savefig('raw_data.png')


def plot_elbow(wcss):
    """
    Plot elbow method for k-means clustering
    :param wcss: List of 13 elements (from n_clusters=3 to n_clusters=15)
    :return: None
    """
    plt.figure()
    K = range(3, 16)
    plt.plot(K, wcss, 'bx-')
    plt.savefig('elbow.png')


def visualize_cluster_results(train_dict, n_clusters):
    """
    :param train_dict: dictionary with keys a1, a2, ..., a6 where the values are the clusters
    :param n_clusters: int (optional for using)
    :return:
    """

    """ TODO: 
        Your code goes here
    """
    fig, ax = plt.subplots(1, figsize=(7, 7))
    for i in range(len(train_dict)):
        stepfunc = train_dict[f'a{i + 1}']
        stepfunc = np.array(stepfunc)
        stepfunc = stepfunc.flatten() + 1
        ax.plot(stepfunc, label=f'specimen a{i + 1}')

    ax.set_xlabel('time steps')
    ax.set_ylabel('clusters')
    ax.legend(loc='upper left')
    fig.suptitle('Sojourn time distributions')
    plt.savefig('Sojourn_distributions.png')


def plot_distributions(sjrn_t, n_clusters):
    plt.rcParams.update({'font.size': 14})
    exit = False
    if n_clusters < 4:
        cols = n_clusters
    else:
        cols = 4
    fig, ax = plt.subplots(
        nrows=int(math.ceil(n_clusters / cols)),
        ncols=cols, figsize=(25, 30),
        squeeze=False)
    indx = 0
    for i in range(int(math.ceil(n_clusters/cols))):
        for j in range(cols):
            sns.kdeplot(sjrn_t[:, indx], ax=ax[i, j])
            ax[i, j].set_title(f'Cluster {indx + 1}')
            ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            indx += 1
            if indx == n_clusters:
                exit = True
                break
        if exit:
            break
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('distributions.png')


def plot_fitted_and_initial_distributions(sjrn_t, n_clusters, dist_param_list):
    exit = False
    if n_clusters < 4:
        cols = n_clusters
    else:
        cols = 4
    fig, ax = plt.subplots(
        nrows=int(math.ceil(n_clusters / cols)),
        ncols=cols, figsize=(25, 20),
        squeeze=False)
    indx = 0
    for i in range(int(math.ceil(n_clusters/cols))):
        for j in range(cols):
            sns.kdeplot(sjrn_t[:, indx], ax=ax[i, j])

            c, loc, scale = dist_param_list[i]
            samples = ss.weibull_min.rvs(c, loc=loc, scale=scale, size=1000)
            sns.kdeplot(samples, ax=ax[i, j])
            ax[i, j].set_title(f'Cluster {indx + 1}')
            ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            indx += 1
            if indx == n_clusters:
                exit = True
                break
        if exit:
            break
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('fitted_initial_distributions.png')


def train(train_dict, data_dict, n_clusters, cluster_method, linkage):
    # use experiments a1 - a6 for training, a7-a8 for testing (named as te_a1, te_a2)

    x_train = []
    if cluster_method == 'kmeans':
        wcss = []
        for i in range(len(train_dict)):
            x_train.extend(data_dict[f'a{i + 1}']['strains'])
    else:  # in Agglomerative all samples should be used
        for i in range(len(data_dict)):
            if i < 6:
                x_train.extend(data_dict[f'a{i + 1}']['strains'])
            else:
                x_train.extend(data_dict[f'te_a{i + 1 - 6}']['strains'])
    x_train = np.array(x_train).squeeze().reshape(-1, 1)

    if cluster_method == 'kmeans':
        x_train = np.sort(x_train.reshape(-1))

        # Clustering using k-means, to check the optimal n_clusters with the elbow method (WCSS)
        # multiple k-means should be run in a loop, then you can plot the elbow (see plot_elbow() function defined above).
        # When you finish don't forget to remove the loop and comment again the plot_elbow(wcss) call...

        cluster_model = KMeans(n_clusters=n_clusters, n_init=10).fit(x_train.reshape(-1, 1))
        wcss.append(cluster_model.inertia_)
        # plot_elbow(wcss)

        _, indx = np.unique(cluster_model.labels_, return_index=True)
        lookup_labels = [cluster_model.labels_[i] for i in sorted(indx)]
    else:
        # Clustering using Agglomerative
        # Initialize and fit the clustering model using sklearn
        # The hyperparameters to use are the number of clusters and the linkage method
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_model = cluster_model.fit(x_train)
        lookup_labels = cluster_model.labels_

        """ TODO: 
            Your code goes here
        """

    return cluster_model, lookup_labels, n_clusters


def predict(cluster_model, lookup_labels, data_dict, cluster_method, train):
    if train:
        clusters_dict = {f'a{i + 1}': [] for i in range(6)}
    else:
        clusters_dict = {f'te_a{i + 1}': [] for i in range(2)}

    for i in range(len(data_dict)):
        if cluster_method == 'kmeans':
            if train:
                pred = cluster_model.predict(
                    np.array(data_dict[f'a{i + 1}']['strains']).squeeze().reshape(-1, 1))
            else:
                pred = cluster_model.predict(
                    np.array(data_dict[f'te_a{i + 1}']['strains']).squeeze().reshape(-1, 1))
            labels = []
            for j in range(len(pred)):
                labels.append(np.where(lookup_labels == pred[j])[0])
            if train:
                clusters_dict[f'a{i + 1}'].extend(sorted(labels))
            else:
                clusters_dict[f'te_a{i + 1}'].extend(sorted(labels))
        else:
            if train:
                labels = cluster_model.fit_predict(
                    np.array(data_dict[f'a{i + 1}']['strains']).squeeze().reshape(-1, 1))
                clusters_dict[f'a{i + 1}'].extend(sorted(labels))
            else:
                labels = cluster_model.fit_predict(
                    np.array(data_dict[f'te_a{i + 1}']['strains']).squeeze().reshape(-1, 1))
                clusters_dict[f'te_a{i + 1}'].extend(sorted(labels))

    return clusters_dict


def sojourn_times(labels_dict, n_clusters, time_steps, train):
    """
    :param labels_dict: dictionary of cluster labels for each specimen
    For training, labels_dict has keys a1, a2, a3, a4, a5, a6, for testing it has keys te_a1, te_a2
    :param n_clusters: int --> number of clusters
    :param time_steps: int --> !!! don't forget the sojourn times should be measured in cycles not in steps !!!
    :param train: bool --> whether we are in training or testing phase (for naming the dictionary keys accordingly)
    :return: sojourn times array of shape [specimens, n_clusters] where specimens=train specimens for training, specimens=testing specimens for testing
    """

    # shape of sojourn times :  [specimens, n_clusters]
    sjrn_t = np.zeros((len(labels_dict), n_clusters))
    for i in range(len(labels_dict)):
        if train:
            labels = labels_dict[f'a{i + 1}']
        else:
            labels = labels_dict[f'te_a{i + 1}']
        for j in range(n_clusters):
            sjrn_t[i, j] = labels.count(j)
        sjrn_t[i, :] *= time_steps  # rescale to cycles
    sjrn_t[np.where(sjrn_t == 0)] = time_steps
    return sjrn_t


def fit_distribution_per_cluster(sojourn_times):
    dist = ss.weibull_min
    dist_param = []
    for i in range(sojourn_times.shape[1]):
        # store a tuple of parameters in order (if applicable): (shape, loc, scale)
        dist_param.append(dist.fit(sojourn_times[:, i], floc=0))
    return dist_param


def create_pdf_samples(dist_param_list, n_clusters):
    samples = []
    for i in range(n_clusters):
        c, loc, scale = dist_param_list[i]
        samples.append(np.sort(ss.weibull_min.rvs(c, loc=loc, scale=scale, size=10)))
    return samples


def rul_estimation(samples, data_clusters, sojourn_times, time_steps, n_clusters):
    """
    :param samples: List of np.array, the samples generated from the fitted distributions for each cluster
    :param data_clusters: dictionary of cluster labels for each specimen for testing phase (has keys te_a1, te_a2)
    :param sojourn_times: np.array of shape [test specimens, n_clusters]
    :param time_steps: int
    :param n_clusters: int
    :return: dictionary of rul estimations for each specimen
    """

    ruls = {f'te_a{i + 1}': {'mean_rul': []} for i in range(len(data_clusters))}

    for i in range(len(ruls)):
        prev = 0
        for j in range(n_clusters):
            time = np.arange(0, sojourn_times[i, j], time_steps)
            tot_rul = np.zeros((samples[j].shape[0]))
            rul = np.zeros((time.shape[0]))
            for k in range(j, n_clusters):
                tot_rul += samples[k]
            tot_mean = tot_rul.mean()
            for k in range(time.shape[0]):
                rul[k] = tot_mean - time[k]

            ruls[f'te_a{i + 1}']['mean_rul'].extend(rul)
            prev += sojourn_times[i, j]
    return ruls


def visualize_rul(test_dict, pred_rul_dict, time_steps):
    """
    Function to visualize the test RUL estimations for the specimens te_a1 and te_a2.
    Feel free to use any variable you want from the main function.
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(23, 35), squeeze=False)
    indx = 0
    for i in range(1):
        for j in range(2):
            true_rul = np.array(test_dict[f'te_a{indx + 1}']['ruls']).squeeze()
            pred_rul = np.array(pred_rul_dict[f'te_a{indx + 1}']['mean_rul']).squeeze()
            x = np.arange(0, true_rul[0] + time_steps, time_steps)
            if pred_rul.shape[0] != true_rul.shape[0]:
                diff = pred_rul.shape[0] - true_rul.shape[0]
                pred_rul = np.delete(pred_rul, obj=-np.arange(diff))
            ax[i, j].plot(x, true_rul, linestyle='dashed', color='black', label='True RUL')
            ax[i, j].plot(x, pred_rul, color='blue')
            ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax[i, j].set_xlabel('time steps')
            ax[i, j].set_ylabel('RUL')
            ax[i, j].set_title(f'Test specimen te_{indx + 1}')
            indx += 1
            ax[i, j].legend()
    fig.suptitle('Mean RUL predictions for train-test specimens')
    plt.savefig('rul_predictions.png')
    return None


def rul_estimation_with_uncertainty(samples, data_clusters, sojourn_times, time_steps, n_clusters):
    """

    :param samples: List of np.array, the samples generated from the fitted distributions for each cluster
    :param data_clusters: dictionary of cluster labels for each specimen for testing phase (has keys te_a1, te_a2)
    :param sojourn_times: np.array of shape [test specimens, n_clusters]
    :param time_steps: int
    :param n_clusters: int
    :return: dictionary of rul estimations for each specimen for mean, min and max of RUL corresponding to 95% confidence interval
    """

    # Hint: use scipy.stats.bootstrap (read the docs!) to estimate the confidence intervals by feeding the samples as input

    ruls = {f'te_a{i + 1}': {'mean_rul': [], 'min_rul': [], 'max_rul': []}
            for i in range(len(data_clusters))}

    for i in range(len(ruls)):
        """ TODO: 
            Your code goes here
        """

    return ruls


def visualize_rul_with_uncertainty(test_dict, pred_rul_dict, time_steps):
    """
    Function to visualize the test RUL estimations for the specimens te_a1 and te_a2 with uncertainty.
    Feel free to use any variable you want from the main function.
    """

    """ TODO: OPTIONAL
        Your code goes here:
    """

    return None


def mse_loss(test_dict, pred_rul_dict):
    """
    :param test_dict: Nested dictionary with true RUL data
    :param pred_rul_dict: Nested dictionary with predicted RUL data
    :return: MSE loss
    """
    # Hint: Check the format of the dictionaries (nested dicts) and convert it to numpy array)
    actual_1 = np.array(test_dict['te_a1']['ruls'])
    mean_1 = np.array(pred_rul_dict['te_a1']['mean_rul'])
    actual_2 = np.array(test_dict['te_a2']['ruls'])
    mean_2 = np.array(pred_rul_dict['te_a2']['mean_rul'])
    # se1 = 0.0
    # se2 = 0.0
    # for i in range(len(mean_1)):
    #     se1_i = (actual_1[i] - mean_1[i])**2
    #     se1 = se1 + se1_i
    # for i in range(len(mean_2)):
    #     se2_i = (actual_2[i] - mean_2[i])**2
    #     se2 = se2 + se2_i
    se1 = np.sum((actual_1 - mean_1)**2)
    se2 = np.sum((actual_2 - mean_2)**2)
    mse = ((se1 / len(mean_1))**0.5 + (se2 / len(mean_2))**0.5) / 2

    """ TODO: 
        Your code goes here
    """
    return mse


if __name__ == '__main__':
    print("Code running...\n")

    time_steps = 50  # every 50 seconds is one step, it's easier to use time steps, then convert to seconds for the results phase

    cluster_algorithm = ['kmeans', 'agglomerative']

    # Part 1: Clustering
    print('\n Part 1: Clustering')
    print('\n 1. Load two json files, the train.json and test.json for the training and testing data.'
          '\n 6 specimens for training and 2 for testing. Then, visualize the dataset.')
    input("\n Proceed ? Press any button to continue... ")

    # load from json
    # Train dictionary with keys: 'a1', 'a2', 'a3', 'a4', 'a5', 'a6' for the first 6 specimens
    with open('data/train.json', 'r') as fp:
        train_dict = json.load(fp)
    # Test dictionary with keys: 'te_a1', 'te_a2' for the remaining 2 specimens
    with open('data/test.json', 'r') as fp:
        test_dict = json.load(fp)

    # Stack together train, test dictionaries for utilization in the next steps
    data_dict = {}
    data_dict.update(train_dict)
    data_dict.update(test_dict)

    visualize_data(train_dict, train=True)
    visualize_data(test_dict, train=False)

    # 2. Create clusters using a training set
    print('\n 2. Create clusters using a training set')
    input("\n Proceed ? Press any button to continue... ")
    while True:
        choose_algo = input(
            '\n 2. Please choose a clustering algorithm from [K-means, Agglomerative]. Type "0" for Kmeans, "1" for Agglomerative : ')
        if choose_algo in ['0', '1']:
            break
        else:
            print('\n Wrong value given. Please choose one of the available clustering algorithms ( type 0 or 1)')

    cluster_method = cluster_algorithm[int(choose_algo)]

    if int(choose_algo) == 0:
        while True:
            n_clusters = input(
                '\n 3. Please enter the number of clusters for K-means in range [3, 15] : ')
            if 3 <= int(n_clusters) <= 15:
                break
            else:
                print(
                    '\n Number of clusters is out of the given range. Please make sure the number you typed belongs to the desired range')
        n_clusters = int(n_clusters)
        linkage = None
    else:
        while True:
            n_clusters = input(
                '\n 3. Please enter the number of clusters for Agglomerative in range [3, 15] : ')
            if 3 <= int(n_clusters) <= 15:
                break
            else:
                print(
                    '\n Number of clusters is out of the given range. Please make sure the number you typed belongs to the desired range')
        n_clusters = int(n_clusters)
        while True:
            linkage = input(
                "\n Please define the hyperparameter linkage for Agllomerative algorithm. "
                "Choose between {'ward', 'complete', 'average'} : ")
            if linkage in ['ward', 'complete', 'average']:
                break
            else:
                print(
                    '\n The given linkage method is not available. Please make sure you typed the method correctly.')

    # Build the cluster model
    cluster_model, lookup_labels, n_clusters = train(
        train_dict, data_dict, n_clusters, cluster_method, linkage)

    # Part 2: Sojourn time distributions
    print('\n Part 2: Sojourn time distributions')
    input("\n Proceed ? Press any button to continue... ")

    # Store on dictionaries the corresponding labels where the raw data were clustered
    train_labels_dict = predict(cluster_model, lookup_labels,
                                train_dict, cluster_method, train=True)
    test_labels_dict = predict(cluster_model, lookup_labels, test_dict, cluster_method, train=False)

    # 1. Degradation history for each training specimen in terms of label (cluster) sequence
    print('\n 1. Degradation history for each training specimen in terms of label (cluster) sequence')

    visualize_cluster_results(train_labels_dict, n_clusters=n_clusters)

    # 2. Identifying sojourn times at each cluster for each training specimen
    print('\n 2. Identifying sojourn times at each cluster for each training specimen')
    train_sojourn_time_arr = sojourn_times(train_labels_dict, n_clusters, time_steps, train=True)

    # 3.Plot probability distributions on sojourn times
    print('\n 3. Fitting probability distributions on sojourn times')
    plot_distributions(train_sojourn_time_arr, n_clusters)

    # 5. Output
    print('\n 5. Output')
    # fit the chosen family of distributions to the sojourn times
    train_dist_param_list = fit_distribution_per_cluster(train_sojourn_time_arr)
    # generate samples of the fitted distributions
    samples = create_pdf_samples(train_dist_param_list, n_clusters)

    # 6. Plot at the same graph the fitted distributions and the initial pdf of sojourn times
    print('\n 6. Plot at the same graph the fitted distributions and the initial pdf of sojourn times')
    plot_fitted_and_initial_distributions(train_sojourn_time_arr, n_clusters, train_dist_param_list)

    # Part 3: Prognostics
    print('\n Part 3: Prognostics')
    input("\n Proceed ? Press any button to continue... ")

    # 1. Use the tesing data to create the test sojourn times
    print('\n 1. Use the tesing data to create the test sojourn times')

    test_sojourn_time_arr = sojourn_times(test_labels_dict, n_clusters, time_steps, train=False)

    # 2. Calculate the RUL for each testing specimen
    print('\n 2. Calculate the RUL for each testing specimen')
    test_pred_rul_dict = rul_estimation(
        samples, test_labels_dict, test_sojourn_time_arr, time_steps, n_clusters)

    # 3. Visualize the RUL for each testing specimen
    print('\n 3. Visualize the RUL for each testing specimen')
    visualize_rul(test_dict, test_pred_rul_dict, time_steps)

    # Part 4: Error calculation
    print('\n 4. Error calculation ')
    input("\n Proceed ? Press any button to continue... ")

    test_mse = mse_loss(test_dict, test_pred_rul_dict)
    print(f"\n MSE loss for testing is {round(test_mse, 4)}")

    # Part 5. Optional task: Write a script able to predict not only the mean RUL but also the related X% confidence intervals (CIs) of a testing specimen
    # and plot the estimated mean RUL values versus the actual RUL (y-axis: mean RUL, 95% CIs, actual RUL/ x-axis: timestep).

    print('\n 5. Optional task: Write a script able to predict not only the mean RUL but also the related X% confidence intervals (CIs) of a testing specimen '
          'and plot the estimated mean RUL values versus the actual RUL (y-axis: mean RUL, 95% CIs, actual RUL/ x-axis: timestep).')

    test_pred_rul_dict = rul_estimation_with_uncertainty(
        samples, test_labels_dict, test_sojourn_time_arr, time_steps, n_clusters)

    visualize_rul_with_uncertainty(test_dict, test_pred_rul_dict, time_steps)
