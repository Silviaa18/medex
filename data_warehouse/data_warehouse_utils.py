import collections
import pickle

import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.metrics.pairwise
import sklearn.mixture
import sklearn.preprocessing
import tqdm

import data_warehouse.redis_rwh as rwh

import misc.parallel as parallel
import misc.utils as utils

import logging
logger = logging.getLogger(__name__)

def get_entity_maps(entity_file):

    entity_df = pd.read_csv(entity_file)

    entity_index_map = {
        entity['name']: i for i, entity in entity_df.iterrows()
    }

    rev_entity_index_map = {
        i: entity['name'] for i, entity in entity_df.iterrows()
    }

    entity_type_map = {
        entity['name']: entity['value_type'] for i, entity in entity_df.iterrows()
    }

    return entity_index_map, rev_entity_index_map, entity_type_map

def unpickle_entity_info(entity_info_file):

    with open(entity_info_file, "rb") as f:
        entity_info = pickle.load(f)

    return entity_info


def get_itemset_names(fim_result, rev_entity_index_map):

    itemset_indices = fim_result[0]
    count = fim_result[1]

    itemset_names = []

    for item in itemset_indices:
        entity_name = rev_entity_index_map[item]
        itemset_names.append(entity_name)
        
    return itemset_names

def get_co_occurrence_matrix(sparse_patient_info):
    
    num_patients = sparse_patient_info.shape[0]
    num_entities = sparse_patient_info.shape[1]
    patient_observed_values = np.zeros(num_patients, dtype=object)

    co_occurrence_matrix = np.zeros((num_entities, num_entities), dtype=int)

    for i in tqdm.tqdm(range(num_patients)):
        patient_observed_values_i = list(sparse_patient_info[i].nonzero()[1])
        num_observed_entities = len(patient_observed_values_i)

        for j in range(num_observed_entities):
            entity_j = patient_observed_values_i[j]
            
            for k in range(i+1, num_observed_entities):
                entity_k = patient_observed_values_i[k]
            
                co_occurrence_matrix[entity_j, entity_k] += 1

                #msg = "Setting [{}, {}]: {}".format(entity_j, entity_k, co_occurrence_matrix[entity_j, entity_k])
                #print(msg)
                
        patient_observed_values[i] = patient_observed_values_i

    return patient_observed_values, co_occurrence_matrix

def get_co_occurrence_graph(co_occurrence_matrix):

    num_entities = co_occurrence_matrix.shape[0]
    co_occurrence_graph = DimacsGraph(num_entities, weighted=True, directed=False)

    for i in tqdm.tqdm(range(num_entities)):
        for j in range(i+1, num_entities):
            
            edge_weight = co_occurrence_matrix[i, j]
            
            if edge_weight != 0:
                co_occurrence_graph.addEdge(i, j, edge_weight)

    return co_occurrence_graph

def clean_entity_names(entity_names):
    """ Remove prefixes and similar from the entity names
    
    Parameters
    ----------
    entity_names: pd.Series
        The raw entity names
        
    Returns
    -------
    cleaned_entity_names: pd.Series
        The entity names with prefixes and such removed
    """
    entity_names = entity_names.str.replace("diagnostik.labor.", "")
    return entity_names

ALLOWED_MISSING = {
    "drop",
    "mean",
    "median",
    "most_frequent"
}

def _retrieve_numeric_fieds_df(entities, r, standardize=True, missing='drop', min_max_filter=None):
    """ Retrieve patient records from the db and preprocess as specified.

    Paramters
    ---------
    entities: list of strings
        The names of the numeric entities

    r : redis.StrictRedis
        The redis database connection

    standardize : bool
        Whether to standardize the values before clustering

    missing : string
        How to handle missing data. The available values are:
            * "drop" : do not consider rows with missing data
            * "mean", "median", "most_frequent": these values are passed to an
                sklearn.preprocessing.Imputer. Please see its documentation for
                more details.

    Returns
    -------
    patient_df: pd.DataFrame
        The data frame containing entity values for each patient

    scaler: sklearn.preprocessing.StandardScaler, or None
        The scaler to standardize the data, or None if the data is not 
        standardized. The idea is that standardized values can be transformed
        back, and raw values can be transformed, using this.

    """    
    if missing not in ALLOWED_MISSING:
        msg = "Invalid \"missing\" argument: {}".format(missing)
        raise ValueError(msg)


    df, error = rwh.get_joined_numeric_values(entities, r, min_max_filter)
    if error:
        # df, scaler
        return None, None, error

    if missing == "drop":
        df = df.dropna()
        X = df[entities].values
    else:
        X = df[entities].values
        imp = sklearn.preprocessing.Imputer(strategy=missing)
        X = imp.fit_transform(X)

    scaler = None
    if standardize:
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

    df[entities] = X
    return df, scaler, None


def cluster_numeric_fields(entities, r, standardize=True, missing='drop', min_max_filter=None):
    """ Cluster the given numeric entities using a DP-GMM.

    Paramters
    ---------
    entities: list of strings
        The names of the numeric entities

    r : redis.StrictRedis
        The redis database connection

    standardize : bool
        Whether to standardize the values before clustering

    missing : string
        How to handle missing data. The available values are:
            * "drop" : do not consider rows with missing data
            * "mean", "median", "most_frequent": these values are passed to an
                sklearn.preprocessing.Imputer. Please see its documentation for
                more details.

    Returns
    -------
    m : sklearn.mixture.BayesianGaussianMixture
        The fit mixture model

    label_uses : dict
        A dictionary mapping from each label to the number of items with that 
        label.

    patient_df: pd.DataFrame
        The data frame containing entity values for each patient
    """
    import misc.math_utils as math_utils

    df, scaler, error = _retrieve_numeric_fieds_df(
        entities, 
        r, 
        standardize=standardize, 
        missing=missing,
        min_max_filter=min_max_filter
    )
    if error:
        # cluster_data, cluster_labels, df
        return None, None, None,  error
    X = df[entities].values
    
    # make sure we do not use too many components
    # otherwise, sklearn will complain
    n_components = min(100, X.shape[0]-1)

    cluster_data = math_utils.fit_bayesian_gaussian_mixture(
        X, 
        n_components=n_components,
        max_iter=10000,
        reg_covar=0,
        mean_precision_prior=0.8,
        weight_concentration_prior_type="dirichlet_process", 
        init_params='kmeans',
        seed=8675309
    )

    clusters = cluster_data.predict(X)

    df['cluster'] = clusters

    cluster_labels = collections.defaultdict(int)
    for l in clusters:
        cluster_labels[l] += 1

    if standardize:
        for i, means in enumerate(cluster_data.means_):
            cluster_data.means_[i] = scaler.inverse_transform(means)

    return cluster_data, cluster_labels, df, None

def _get_patient_categorical_rep(row, categorical_entities):
    ret = {}
    
    for categorical_entity in categorical_entities:
        vals = row[categorical_entity]
        
        # make sure this is a list of values, which is what
        # we expect from rwh.get_joined_categorical_values
        if not utils.is_sequence(vals):
            continue
        
        for val in vals:
            key = "{}.{}".format(categorical_entity, val)
            ret[key] = 1
    return ret

def cluster_categorical_entities(entities, r, eps=0.1, min_samples=5, 
        seed=8675309):
    """ Cluster patients based on the given categorical entities using DBSCAN.
    The Jaccard distance is used for clustering.

    Paramters
    ---------
    entities: list of strings
        The names of the numeric entities

    r: redis.StrictRedis
        The redis database connection

    eps: float
        The maximum distance between two samples to clsuter them together.
        Please see sklearn.cluster.DBSCAN for more details.

    min_samples: int
        The minimum number of samples in each cluster

    seed: int
        The random seed

    Returns
    -------
    cluster_category_values: dict
        Mapping from cluster label to the count of each category value observed
        for patients in that cluster

    patient_np: 2-dimensional (binary) np.array
        The indicator matrix of each (entity,value) for each patient

    category_values: list of pairs of strings
        All observed (category,value) pairs

    label_uses : dict
        A dictionary mapping from each label to the number of items with that 
        label.

    patient_df: pd.DataFrame
        The data frame containing entity values for each patient
    """
    # pull the values from the database
    cat_df, error = rwh.get_joined_categorical_values(entities, r) #.dropna()
    if error:
        # cluster_category_values, cat_rep_np, category_values, label_uses, cat_df
        return None, None, None, None, None, error
    # the error is here. cat_rep returns dicts with empty values
    # get the binary representation
    cat_rep = parallel.apply_df_simple(
        cat_df, 
        _get_patient_categorical_rep, 
        entities
    )
    
    # convert it to a numpy 2-d array; each row is a patient
    cat_rep_df = pd.DataFrame(cat_rep)
    category_values = sorted(cat_rep_df.columns)
    cat_rep_np = cat_rep_df.fillna(0).values

    # calculate the jaccard distances between all patients
    distance = sklearn.metrics.pairwise.pairwise_distances(
        cat_rep_np, 
        metric='jaccard'
    )

    # cluster the patients based on the jaccard distances
    db = sklearn.cluster.DBSCAN(
        eps=eps, 
        min_samples=min_samples, 
        metric='precomputed'\
    )
    db.fit(distance)

    # add the clustering information to the 
    cat_df['cluster'] = db.labels_

    # collect the usage information
    unique_labels = set(db.labels_)
    label_uses = {}
    cluster_category_values = {}

    for l in unique_labels:
        m_class = (db.labels_ == l)
        label_uses[l] = np.sum(m_class)
        cluster_category_values[l] = cat_rep_df[m_class].count()

    return cluster_category_values, cat_rep_np, category_values, label_uses, cat_df, None



