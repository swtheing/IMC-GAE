"""MovieLens dataset"""
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
from utils import to_etype_name

import pickle as pkl
import h5py
import pdb
import random
from scipy.sparse import linalg
from data_utils import load_data, map_data, download_dataset
from sklearn.metrics import mean_squared_error
from math import sqrt
from bidict import bidict

_urls = {
    'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-10m' : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
}

_paths = {
    'flixster' : './raw_data/flixster/training_test_dataset.mat',
    'douban' : './raw_data/douban/training_test_dataset.mat',
    'yahoo_music' : './raw_data/yahoo_music/training_test_dataset.mat',
    'ml-100k' : './raw_data/ml-100k/',
    'ml-1m' : './raw_data/ml-1m/',
    'ml-10m' : './raw_data/ml-10M100K/'
}

READ_DATASET_PATH = get_download_dir()
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csr_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


class DataSetLoader(object):
    """
    TODO(minjie): make this dataset more general

    The dataset stores MovieLens ratings in two types of graphs. The encoder graph
    contains rating value information in the form of edge types. The decoder graph
    stores plain user-movie pairs in the form of a bipartite graph with no rating
    information. All graphs have two types of nodes: "user" and "movie".

    The training, validation and test set can be summarized as follows:

    training_enc_graph : training user-movie pairs + rating info
    training_dec_graph : training user-movie pairs
    valid_enc_graph : training user-movie pairs + rating info
    valid_dec_graph : validation user-movie pairs
    test_enc_graph : training user-movie pairs + validation user-movie pairs + rating info
    test_dec_graph : test user-movie pairs

    Attributes
    ----------
    train_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for training.
    train_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for training.
    train_labels : torch.Tensor
        The categorical label of each user-movie pair
    train_truths : torch.Tensor
        The actual rating values of each user-movie pair
    valid_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for validation.
    valid_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for validation.
    valid_labels : torch.Tensor
        The categorical label of each user-movie pair
    valid_truths : torch.Tensor
        The actual rating values of each user-movie pair
    test_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for test.
    test_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for test.
    test_labels : torch.Tensor
        The categorical label of each user-movie pair
    test_truths : torch.Tensor
        The actual rating values of each user-movie pair
    user_feature : torch.Tensor
        User feature tensor. If None, representing an identity matrix.
    movie_feature : torch.Tensor
        Movie feature tensor. If None, representing an identity matrix.
    possible_rating_values : np.ndarray
        Available rating values in the dataset

    Parameters
    ----------
    name : str
        Dataset name. Could be "ml-100k", "ml-1m", "ml-10m"，"flixster","","douban","yahoo_music"
    device : torch.device
        Device context
    mix_cpu_gpu : boo, optional
        If true, the ``user_feature`` attribute is stored in CPU
    use_one_hot_fea : bool, optional
        If true, the ``user_feature`` attribute is None, representing an one-hot identity
        matrix. (Default: False)
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    test_ratio : float, optional
        Ratio of test data
    valid_ratio : float, optional
        Ratio of validation data

    """
    def __init__(self, name, device, mix_cpu_gpu=False,
                 use_one_hot_fea=True, symm=True,
                 test_ratio=0.1, valid_ratio=0.1,sparse_ratio = 0):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        self._dir = os.path.join(_paths[self._name])
        if self._name in ['ml-100k', 'ml-1m', 'ml-10m']:
            # download and extract
            download_dir = get_download_dir()
            print("download_dir: ", download_dir)
            zip_file_path = '{}/{}.zip'.format(download_dir, name)
            download(_urls[name], path=zip_file_path)
            extract_archive(zip_file_path, '{}/{}'.format(download_dir, name))
            if name == 'ml-10m':
                root_folder = 'ml-10M100K'
            else:
                root_folder = name
            self._dir = os.path.join(download_dir, name, root_folder)
            print("Starting processing {} ...".format(self._name))
            self._load_raw_user_info()
            self._load_raw_movie_info()
            print('......')
            if self._name == 'ml-100k':
                self.all_train_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\t')
                self.test_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\t')
                self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
            elif self._name == 'ml-1m' or self._name == 'ml-10m':
                self.all_rating_info = self._load_raw_rates(os.path.join(self._dir, 'ratings.dat'), '::')
                num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
                shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
                self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
                self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]
            else:
                raise NotImplementedError
            print('......')
            num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
            shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
            self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
            self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)
        elif self._name in ['flixster', 'douban', 'yahoo_music']:
            self._dir = os.path.join(_paths[self._name])
            rating_map = None
            testing = True
            post_rating_map = None
            data_name = self._name
            (
            u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
            val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
            test_v_indices, class_values, user_nodes, item_nodes
            ) = self.load_data_monti(data_name, testing, rating_map, post_rating_map)
            train_labels = [class_values[i] for i in train_labels]
            test_labels = [class_values[i] for i in test_labels]
            self.all_train_rating_info = self.trans_loader(train_labels, train_u_indices, train_v_indices)
            self.test_rating_info = self.trans_loader(test_labels, test_u_indices, test_v_indices)
            self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
            print("train size {} , test size {} , all size {}".format(len(self.all_train_rating_info), len(self.test_rating_info), len(self.all_rating_info)))
            print("train u_size {} v_size {}; test u_size {} v_size {}; ".format(len(train_u_indices), len(train_v_indices), len(test_u_indices), len(test_v_indices)))

            print('......')
            num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
            shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
            #shuffled_idx = np.arange(self.all_train_rating_info.shape[0])
            self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]

            self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)
            self.max_l = np.max(self.possible_rating_values)
            self.min_l = np.min(self.possible_rating_values)
            print(self.possible_rating_values)
        else:
            raise NotImplementedError
        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs  : {}".format(self.test_rating_info.shape[0]))

        if self._name in ['ml-100k', 'ml-1m', 'ml-10m']:
            self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                    cmp_col_name="id",
                                                    reserved_ids_set=set(self.all_rating_info["user_id"].values),
                                                    label="user")
            self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                    cmp_col_name="id",
                                                    reserved_ids_set=set(self.all_rating_info["movie_id"].values),
                                                    label="movie")

            # Map user/movie to the global id
            self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
            self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        elif self._name in ['flixster', 'douban', 'yahoo_music']:
            self.global_user_id_map = bidict({})
            self.global_movie_id_map = bidict({})
            # max_uid = 0
            # max_vid = 0
            print("user and item number:")
            # print(user_nodes)
            # print(item_nodes)
            for i in range(len(user_nodes)):
                self.global_user_id_map[user_nodes[i]] = i
            for i in range(len(item_nodes)):
                self.global_movie_id_map[item_nodes[i]] = i
        else:
            raise NotImplementedError

        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)
        ### Generate features
        if use_one_hot_fea:
            self.user_feature = None
            self.movie_feature = None
        else:
            # if mix_cpu_gpu, we put features in CPU
            if mix_cpu_gpu:
                self.user_feature = th.FloatTensor(self._process_user_fea())
                self.movie_feature = th.FloatTensor(self._process_movie_fea())
            else:
                self.user_feature = th.FloatTensor(self._process_user_fea()).to(self._device)
                self.movie_feature = th.FloatTensor(self._process_movie_fea()).to(self._device)

        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user + self.num_movie + 3)
            self.movie_feature_shape = (self.num_movie, self.num_user + self.num_movie + 3)
            if mix_cpu_gpu:
                self.user_feature = th.cat([th.Tensor(list(range(3, self.num_user+3))).reshape(-1, 1), th.zeros([self.num_user, 1])+1, th.zeros([self.num_user, 1])], 1)
                self.movie_feature = th.cat([th.Tensor(list(range(3, self.num_movie+3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1)
                #self.movie_feature = th.cat([th.Tensor(list(range(self.num_user+3, self.num_user + self.num_movie + 3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1)
            else:
                self.user_feature = th.cat([th.Tensor(list(range(3, self.num_user+3))).reshape(-1, 1), th.zeros([self.num_user, 1])+1, th.zeros([self.num_user, 1])], 1).to(self._device)
                self.movie_feature = th.cat([th.Tensor(list(range(self.num_user+3, self.num_user + self.num_movie + 3))).reshape(-1, 1), th.ones([self.num_movie, 1])+1, th.zeros([self.num_movie, 1])], 1).to(self._device)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.movie_feature_shape = self.movie_feature.shape
        #print(self.user_feature.shape)
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)

        def _make_labels(ratings):
            labels = th.LongTensor(np.searchsorted(self.possible_rating_values, ratings)).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values, add_support=True)
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            _npairs(self.train_enc_graph)))
        print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
            self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
            _npairs(self.valid_enc_graph)))
        print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
            self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
            _npairs(self.test_enc_graph)))
        print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
            self.test_dec_graph.number_of_edges()))

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_movie_R = np.zeros((self._num_user, self._num_movie), dtype=np.float32)
        user_movie_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        #assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)
            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user,)
                movie_cj = th.ones(self.num_movie,)
            graph.nodes['user'].data.update({'ci' : user_ci, 'cj' : user_cj})
            graph.nodes['movie'].data.update({'ci' : movie_ci, 'cj' : movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('user', 'rate', 'movie'): g.edges()}, 
                               num_nodes_dict={'user': self.num_user, 'movie': self.num_movie})

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    def load_data_monti(self, dataset, testing=True, rating_map=None, post_rating_map=None, own = False):
        """
        Loads data from Monti et al. paper.
        if rating_map is given, apply this map to the original rating matrix
        if post_rating_map is given, apply this map to the processed rating_mx_train without affecting the labels
        """

        if not own:
            path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'
            M = load_matlab_file(path_dataset, 'M')
            if rating_map is not None:
                M[np.where(M)] = [rating_map[x] for x in M[np.where(M)]]
            print(M.shape)
            Otraining = load_matlab_file(path_dataset, 'Otraining')
            Otest = load_matlab_file(path_dataset, 'Otest')
            num_users = M.shape[0]
            num_items = M.shape[1]
        else:
            path_dataset = 'raw_data/' + dataset + '/douban_train'
        print(path_dataset)
        if dataset == 'flixster':
            Wrow = load_matlab_file(path_dataset, 'W_users')
            Wcol = load_matlab_file(path_dataset, 'W_movies')
            u_features = Wrow
            v_features = Wcol
        elif dataset == 'douban':
            Wrow = load_matlab_file(path_dataset, 'W_users')
            u_features = Wrow
            v_features = np.eye(num_items)
        elif dataset == 'yahoo_music':
            Wcol = load_matlab_file(path_dataset, 'W_tracks')
            u_features = np.eye(num_users)
            v_features = Wcol
        elif dataset == 'own' or dataset == 'all':
            u_features = None
            v_features = None
            rating_train, Train_index, Train_indptr, Train_data, Val_index, Val_indptr, Val_data, Test_index,Test_indptr, Test_data, user_dic, item_dic = load_own_file2(path_dataset)
            Train_indptr = list(np.array(Train_indptr) + len(user_dic))
            Val_indptr = list(np.array(Val_indptr) + len(user_dic))
            Test_indptr = list(np.array(Test_indptr) + len(user_dic))
            class_values = np.array([1, 2, 3, 4, 5])
            num_user = len(user_dic)
            num_item = len(item_dic)
            #print('number of user = ', len(user_dic))
            #print('number of item = ', len(item_dic))
            #print("train_labels:")
            #print(Train_data)
            #print("u_train_idx")
            #print(Train_index)
            #print("v_train_idx")
            #print(Train_indptr)
            #print("test_labels")
            #print(Test_data)
            #print("u_test_idx")
            #print(Test_index)
            #print("v_test_idx")
            #print(Test_indptr)
            #print("class_values")
            #print(class_values)
            return u_features, v_features, rating_train, Train_data, Train_index, Train_indptr, \
                Val_data, Val_index, Val_indptr, Test_data, Test_index, Test_indptr, class_values, num_user, num_item
        elif dataset == 'group':
            rating_train, Train_index, Train_indptr, Train_data, Val_index, Val_indptr, Val_data, Test_index,Test_indptr, Test_data, user_dic, item_dic = load_group_file_rank(path_dataset)
            u_features = range(len(user_dic))
            v_features = range(len(user_dic), len(item_dic)+len(user_dic))
            Train_indptr = list(np.array(Train_indptr) + len(user_dic))
            Val_indptr = list(np.array(Val_indptr) + len(user_dic))
            Test_indptr = list(np.array(Test_indptr) + len(user_dic))
            class_values = np.array([0, 1])
            num_user = len(user_dic)
            num_item = len(item_dic)
            print('number of users = ', len(user_dic))
            print('number of item = ', len(item_dic))
            return u_features, v_features, rating_train, Train_data, Train_index, Train_indptr, \
                Val_data, Val_index, Val_indptr, Test_data, Test_index, Test_indptr, class_values, num_user, num_item

        u_nodes_ratings = np.where(M)[0]
        v_nodes_ratings = np.where(M)[1]
        #print("u_nodes:")
        #print(u_nodes_ratings)
        #print("v_nodes:")
        #print(v_nodes_ratings)
        ratings = M[np.where(M)]
        ''' 
        #Test SVD
        U, s, Vh = linalg.svds(Otraining)
        s_diag_matrix = np.diag(s)
        svd_prediction = np.dot(np.dot(U,s_diag_matrix),Vh)
        prediction_flatten = np.reshape(svd_prediction[Otest.nonzero()], (1,-1))
        test_data_matrix_flatten = Otest[Otest.nonzero()]
        rmse = sqrt(mean_squared_error(prediction_flatten,test_data_matrix_flatten))
        print("SVD rmse:", rmse)
        '''

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)
        u_nodes = u_nodes_ratings
        v_nodes = v_nodes_ratings

        # user_nodes = list(set(u_nodes))
        # item_nodes = list(set(v_nodes))
        # print('number of users = ', len(user_nodes))
        # print('number of items = ', len(item_nodes))

        neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

        # assumes that ratings_train contains at least one example of every rating type
        rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

        labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
        labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
        '''
        for i in range(len(u_nodes)):
            assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])
        '''
        labels = labels.reshape([-1])

        # number of test and validation edges

        num_train = np.where(Otraining)[0].shape[0]
        num_test = np.where(Otest)[0].shape[0]
        num_val = int(np.ceil(num_train * 0.2))
        num_train = num_train - num_val

        pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
        idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

        pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
        idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

        # Internally shuffle training set (before splitting off validation set)
        rand_idx = list(range(len(idx_nonzero_train)))
        np.random.seed(42)
        # np.random.seed(23)
        np.random.shuffle(rand_idx)
        idx_nonzero_train = idx_nonzero_train[rand_idx]
        pairs_nonzero_train = pairs_nonzero_train[rand_idx]

        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:]

        assert(len(test_idx) == num_test)

        val_pairs_idx = pairs_nonzero[0:num_val]
        train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
        test_pairs_idx = pairs_nonzero[num_train + num_val:]

        u_test_idx, v_test_idx = test_pairs_idx.transpose()
        u_val_idx, v_val_idx = val_pairs_idx.transpose()
        u_train_idx, v_train_idx = train_pairs_idx.transpose()

        # create labels
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        if testing:
            u_train_idx = np.hstack([u_train_idx, u_val_idx])
            v_train_idx = np.hstack([v_train_idx, v_val_idx])
            train_labels = np.hstack([train_labels, val_labels])
            # for adjacency matrix construction
            train_idx = np.hstack([train_idx, val_idx])

        class_values = np.sort(np.unique(ratings))

        # make training adjacency matrix
        rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
        '''Note here rating matrix elements' values + 1 !!!'''
        if post_rating_map is None:
            rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
        else:
            rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

        rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))


        if u_features is not None:
            print("user Features:")
            print(u_features)
            u_features = sp.csr_matrix(u_features)
            print("User features shape: " + str(u_features.shape))

        if v_features is not None:
            print("Item Features")
            print(v_features)
            v_features = sp.csr_matrix(v_features)
            print("Item features shape: " + str(v_features.shape))
        print("train_labels: %s" % len(train_labels))
        print(train_labels)
        print("u_train_idx: %s" % len(u_train_idx))
        print(u_train_idx)
        print("v_train_idx: %s" % len(v_train_idx))
        print(v_train_idx)
        print("test_labels: %s" % len(test_labels))
        print(test_labels)
        print("u_test_idx: %s" % len(u_test_idx))
        print(u_test_idx)
        print("v_test_idx: %s" % len(v_test_idx))
        print(v_test_idx)
        print("class_values: %s" % len(class_values))
        print(class_values)

        user_set = set.union(set(u_train_idx), set(u_test_idx))
        item_set = set.union(set(v_train_idx), set(v_test_idx))
        user_nodes = list(user_set)
        item_nodes = list(item_set)
        print('number of users = ', len(user_nodes))
        print('number of items = ', len(item_nodes))

        return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
            val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, user_nodes, item_nodes

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        # print("  -----------------")
        # print("{}: {}(reserved) v.s. {}(from info)".format(label, len(reserved_ids_set),
        #                                                      len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            # print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            # print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _load_raw_rates(self, file_path, sep):
        """In MovieLens, the rates have the following format

        ml-100k
        user id \t movie id \t rating \t timestamp

        ml-1m/10m
        UserID::MovieID::Rating::Timestamp

        timestamp is unix timestamp and can be converted by pd.to_datetime(X, unit='s')

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rating_info : pd.DataFrame
        """
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python',encoding="ISO-8859-1")
        return rating_info

    def _load_raw_user_info(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        ml-1m:
        UserID::Gender::Age::Occupation::Zip-code

        For ml-10m, there is no user information. We read the user id from the rating file.

        Parameters
        ----------
        name : str

        Returns
        -------
        user_info : pd.DataFrame
        """
        if self._name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python',encoding="ISO-8859-1")
        elif self._name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python',encoding="ISO-8859-1")
        elif self._name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(self._dir, 'ratings.dat'), sep='::', header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'movie_id': np.int32, 'ratings': np.float32,
                       'timestamp': np.int64}, engine='python',encoding="ISO-8859-1")
            self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values.astype(np.int32)),
                                     columns=['id'])
        else:
            raise NotImplementedError

    def _process_user_fea(self):
        """

        Parameters
        ----------
        user_info : pd.DataFrame
        name : str
        For ml-100k and ml-1m, the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.
        For ml-10m, there is no user feature and we set the feature to be a single zero.

        Returns
        -------
        user_features : np.ndarray

        """
        if self._name == 'ml-100k' or self._name == 'ml-1m':
            ages = self.user_info['age'].values.astype(np.float32)
            gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
            all_occupations = set(self.user_info['occupation'])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                          dtype=np.float32)
            occupation_one_hot[np.arange(self.user_info.shape[0]),
                               np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
            user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                            gender.reshape((self.user_info.shape[0], 1)),
                                            occupation_one_hot], axis=1)
        elif self._name == 'ml-10m':
            user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
        else:
            raise NotImplementedError
        return user_features

    def _load_raw_movie_info(self):
        """In MovieLens, the movie attributes may have the following formats:

        In ml_100k:

        movie id | movie title | release date | video release date | IMDb URL | [genres]

        In ml_1m, ml_10m:

        MovieID::Title (Release Year)::Genres

        Also, Genres are separated by |, e.g., Adventure|Animation|Children|Comedy|Fantasy

        Parameters
        ----------
        name : str

        Returns
        -------
        movie_info : pd.DataFrame
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [GENRES (19)]]
            For ml-1m and ml-10m, the column name is ['id', 'title'] + [GENRES (18/20)]]
        """
        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self._name == 'ml-100k':
            file_path = os.path.join(self._dir, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python',encoding="ISO-8859-1")
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            file_path = os.path.join(self._dir, 'movies.dat')
            movie_info = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python',encoding="ISO-8859-1")
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(movie_info.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(movie_info['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_info[genre_name] = movie_genres[:, idx]
            self.movie_info = movie_info.drop(columns=["genres"])
        else:
            raise NotImplementedError

    def _process_movie_fea(self):
        """

        Parameters
        ----------
        movie_info : pd.DataFrame
        name :  str

        Returns
        -------
        movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year

        """
        import torchtext

        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        TEXT = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
        embedding = torchtext.vocab.GloVe(name='840B', dim=300)

        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                print('{} cannot be matched, index={}, name={}'.format(title, i, self._name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            title_embedding[i, :] = embedding.get_vecs_by_tokens(TEXT.tokenize(title_context)).numpy().mean(axis=0)
            release_years[i] = float(year)
        movie_features = np.concatenate((title_embedding,
                                         (release_years - 1950.0) / 100.0,
                                         self.movie_info[GENRES]),
                                        axis=1)
        return movie_features

    def trans_loader(self, labels, u_indices, v_indices):
        if len(labels) != len(u_indices) \
                or len(labels) != len(v_indices):
            print("trans_loader: data length error!")
            return None
        size = len(labels)
        data_dict = {
            "user_id":[],
            "movie_id":[],
            "rating":[], 
            "timestamp":[]
        }
        indexs = []
        for i in range(size):
            data_dict["user_id"].append(int(u_indices[i]))
            data_dict["movie_id"].append(int(v_indices[i]))
            data_dict["rating"].append(float(labels[i]))
            data_dict["timestamp"].append(int(999))
            indexs.append(i)
        data_set = pd.DataFrame(data_dict, index=indexs)
        return data_set

if __name__ == '__main__':
    MovieLens("yahoo_music", device=th.device('cpu'), symm=True)
