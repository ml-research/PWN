import torch
import torch.nn as nn
import numpy as np

import time

import model.cwspn.region_graph as region_graph
from model.cwspn.utils import truncated_normal_


# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpnArgs(object):
    def __init__(self):
        self.linear_sum_weights = False
        self.normalized_sums = True
        self.num_sums = 20
        self.num_leaves = 20
        self.param_provider = BasicParamProvider()

        self.gauss_min_sigma = 0.1
        self.gauss_max_sigma = 1.0
        self.dist = 'Gauss'
        self.init_fn = truncated_normal_

        self.gauss_min_mean = None
        self.gauss_max_mean = None


class BasicParamProvider:
    def grab_sum_parameters(self, num_inputs, num_sums):
        return nn.Parameter(torch.Tensor(num_inputs, num_sums))

    def grab_leaf_parameters(self, scope, number, name=None):
        num_inputs = len(scope)
        return nn.Parameter(torch.Tensor(num_inputs, number))


class IndexParamProvider:
    def __init__(self):
        self.current_sum_index = 0
        self.current_leaf_index = 0

        self.sum_params = None
        self.leaf_params = None

    def grab_sum_parameters(self, num_inputs, num_sums):
        index = self.current_sum_index
        self.current_sum_index += num_inputs * num_sums

        return index, (num_inputs, num_sums)

    def grab_leaf_parameters(self, scope, number, name=None):
        num_inputs = len(scope)

        index = self.current_leaf_index
        self.current_leaf_index += num_inputs * number

        return index, (num_inputs, number)

    def get_sum_params(self, param):
        index = param[0]
        num_inputs, num_sums = param[1]

        return self.sum_params[:, index:index + num_inputs * num_sums].reshape((-1, num_inputs, num_sums))

    def get_leaf_params(self, param):
        index = param[0]
        num_inputs, number = param[1]

        return self.leaf_params[:, index:index + num_inputs * number].reshape((-1, num_inputs, number))

class NodeVector(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self):
        raise NotImplemented('NodeVector is an abstract class, it does not implement forward!')

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def init_params(self, init_fn=None):
        pass


class GaussVector(NodeVector):
    def __init__(self, region, args, name, num_dims=0, rv_dim=2):  # rv_dim = 2
        super().__init__(name)
        self.local_size = len(region)
        self.args = args
        self.scope = sorted(list(region))
        self.size = args.num_gauss
        self.num_dims = num_dims
        self.rv_dim = rv_dim
        self.np_means = None
        self.means = self.args.param_provider.grab_leaf_parameters(
            self.scope,
            args.num_gauss * self.rv_dim)

        if args.gauss_min_sigma < args.gauss_max_sigma:
            self.sigma_params = self.args.param_provider.grab_leaf_parameters(
                   self.scope,
                   args.num_gauss * self.rv_dim)
        else:
            self.sigma_params = None

    def forward(self, inputs, marginalized=None):
        means = self.get_means()

        sigma = self.get_sigma()

        # When using the independent setting, we do not model covariances, i.e. the distributions are univariate
        dist = torch.distributions.Independent(torch.distributions.Normal(means, torch.sqrt(sigma)), 1)
        # dist = torch.distributions.Normal(means, torch.sqrt(sigma))

        local_inputs = inputs[:, self.scope]
        if self.rv_dim == 1:
            local_inputs = local_inputs.unsqueeze(-1)

        log_pdf = dist.log_prob(local_inputs.unsqueeze(-2))

        if marginalized is not None:
            marginalized = torch.clamp(marginalized, 0.0, 1.0)
            local_marginalized = marginalized[:, self.scope].unsqueeze(-1)
            log_pdf = log_pdf * (1 - local_marginalized)

        log_pdf = torch.sum(log_pdf, 1)

        return log_pdf

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        if sample:
            raise NotImplementedError
        else:
            my_sample = self.get_means()[case_num]
            # my_sample = sess.run(self.means)
        my_sample = my_sample[:, node_num]
        full_sample = torch.zeros((self.num_dims, self.rv_dim)).to(device)
        full_sample[self.scope] = my_sample
        return full_sample

    def num_params(self):
        result = self.means[1][0] * self.means[1][1]
        if self.sigma_params:
            result += self.sigma_params[1][0] * self.sigma_params[1][1]
        return result

    def get_sigma(self):
        if self.args.gauss_min_sigma < self.args.gauss_max_sigma:
            sigmas = self.args.param_provider.get_leaf_params(self.sigma_params)

            sigma_ratio = torch.sigmoid(sigmas)
            sigma = self.args.gauss_min_sigma + \
                    (self.args.gauss_max_sigma - self.args.gauss_min_sigma) * sigma_ratio

            return sigma.reshape((*sigma.shape[:2], -1, self.rv_dim))

        else:
            return 1.0

    def get_means(self):
        means = self.args.param_provider.get_leaf_params(self.means)

        if self.args.gauss_max_mean:
            assert self.args.gauss_min_mean is not None
            mean_range = self.args.gauss_max_mean - self.args.gauss_min_mean
            means = torch.sigmoid(means) * mean_range + self.args.gauss_min_mean

        return means.reshape((*means.shape[:2], -1, self.rv_dim))

class ProductVector(NodeVector):
    def __init__(self, vector1, vector2, name):
        """Initialize a product vector, which takes the cross-product of two distribution vectors."""
        super().__init__(name)
        self.inputs = [vector1, vector2]
        self.scope = list(set(vector1.scope) | set(vector2.scope))
        assert len(set(vector1.scope) & set(vector2.scope)) == 0
        self.size = vector1.size * vector2.size

    def forward(self, inputs):
        dists1 = inputs[0]
        dists2 = inputs[1]
        batch_size = dists1.shape[0]
        num_dist1 = int(dists1.shape[1])
        num_dist2 = int(dists2.shape[1])

        # we take the outer product via broadcasting, thus expand in different dims
        dists1_expand = dists1.unsqueeze(1)
        dists2_expand = dists2.unsqueeze(2)

        # product == sum in log-domain
        prod = dists1_expand + dists2_expand
        # flatten out the outer product
        prod = prod.view([batch_size, num_dist1 * num_dist2])

        return prod

    def num_params(self):
        return 0

    def sample(self, inputs, seed=None):
        in1_expand = tf.expand_dims(inputs[0], -1)
        in2_expand = tf.expand_dims(inputs[1], -2)

        output_shape = [inputs[0].shape[0], inputs[0].shape[1], (inputs[0].shape[2] * inputs[1].shape[2])]

        result = tf.reshape(in1_expand + in2_expand, output_shape)
        return result

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        row_num = node_num // self.inputs[0].size
        col_num = node_num % self.inputs[0].size
        result1 = self.inputs[0].reconstruct(max_idxs, col_num, case_num, sample)
        result2 = self.inputs[1].reconstruct(max_idxs, row_num, case_num, sample)
        return result1 + result2

class SumVector(NodeVector):
    def __init__(self, prod_vectors, num_sums, args, name=""):
        super().__init__(name)
        self.inputs = prod_vectors  # Python list, so no submodules!
        self.size = num_sums
        self.args = args
        self.scope = self.inputs[0].scope

        for inp in self.inputs:
            assert set(inp.scope) == set(self.scope)

        num_inputs = sum([v.size for v in prod_vectors])
        self.params = self.args.param_provider.grab_sum_parameters(
            num_inputs,
            self.size
        )

        self.max_child_idx = None

    def forward(self, inputs):
        params = self.args.param_provider.get_sum_params(self.params)

        if self.args.linear_sum_weights:
            if self.args.normalized_sums:
                weights = torch.softmax(params, 0)
            else:
                weights = params ** 2
        else:
            if self.args.normalized_sums:
                weights = torch.log_softmax(params, 0)
            else:
                weights = params

        prods = torch.cat(inputs, 1)
        if self.args.linear_sum_weights:
            sums = torch.log(torch.matmul(torch.exp(prods), weights[0]))
        else:
            child_values = prods.unsqueeze(-1) + weights
            self.max_child_idx = torch.argmax(child_values, dim=1)
            sums = torch.logsumexp(child_values, 1)
        return sums

    def reconstruct(self, max_idxs, node_num, case_num, sample):
        my_max_idx = max_idxs[self.name][case_num, node_num]

        for inp_vector in self.inputs:
            if my_max_idx < inp_vector.size:
                return inp_vector.reconstruct(max_idxs, my_max_idx, case_num, sample)

            my_max_idx -= inp_vector.size


    def sample(self, inputs, seed=None):
        inputs = tf.concat(inputs, 2)
        logits = tf.transpose(self.weights[0])
        dist = dists.Categorical(logits=logits)

        indices = dist.sample([inputs.shape[0]], seed=seed)
        indices = tf.reshape(tf.tile(indices, [1, inputs.shape[1]]), [inputs.shape[0], self.size, inputs.shape[1]])
        indices = tf.transpose(indices, [0, 2, 1])

        others = tf.meshgrid(tf.range(inputs.shape[1]), tf.range(inputs.shape[0]), tf.range(self.size))

        indices = tf.stack([others[1], others[0], indices], axis=-1)

        result = tf.gather_nd(inputs, indices)
        return result

    def num_params(self):
        return self.params[1][0] * self.params[1][1]


class RatSpn(nn.Module):
    def __init__(self, num_classes, region_graph, args=SpnArgs(), name=None):
        super().__init__()
        if name is None:
            name = str(id(self))
        self.name = name
        self._region_graph = region_graph
        self.args = args
        self.num_classes = num_classes

        # dictionary mapping regions to tensor of sums/input distributions
        self._region_distributions = dict()
        # dictionary mapping regions to tensor of products
        self._region_products = dict()

        self.vector_list = nn.ModuleList()
        self.output_vector = None

        # make the SPN...
        self.num_dims = region_graph.get_num_items()
        self._make_spn_from_region_graph()
        # self.init_params(init_fn=args.init_fn)

        self.num_dims = len(self.output_vector.scope)

    def _make_spn_from_region_graph(self):
        """Build a RAT-SPN."""

        rg_layers = self._region_graph.make_layers()
        self.rg_layers = rg_layers

        # make leaf layers
        self.vector_list.append(nn.ModuleList())
        for i, leaf_region in enumerate(rg_layers[0]):
            if self.args.dist == 'Gauss':
                name = '{}_gauss_{}'.format(self.name, i)
                leaf_vector = GaussVector(leaf_region, self.args, name, num_dims=self.num_dims)
            elif self.args.dist == 'Bernoulli':
                name = '{}_bernoulli_{}'.format(self.name, i)
                leaf_vector = BernoulliVector(leaf_region, self.args, name, num_dims=self.num_dims)
            self.vector_list[-1].append(leaf_vector)
            self._region_distributions[leaf_region] = leaf_vector

        def add_to_map(given_map, key, item):
            existing_items = given_map.get(key, [])
            given_map[key] = existing_items + [item]

        # make sum-product layers
        ps_count = 0
        for layer_idx in range(1, len(rg_layers)):
            self.vector_list.append(nn.ModuleList())
            if layer_idx % 2 == 1:
                partitions = rg_layers[layer_idx]
                for i, partition in enumerate(partitions):
                    input_regions = list(partition)
                    input1 = self._region_distributions[input_regions[0]]
                    input2 = self._region_distributions[input_regions[1]]
                    vector_name = '{}_prod_{}_{}'.format(self.name, layer_idx, i)
                    prod_vector = ProductVector(input1, input2, vector_name)
                    self.vector_list[-1].append(prod_vector)

                    resulting_region = tuple(sorted(input_regions[0] + input_regions[1]))
                    add_to_map(self._region_products, resulting_region, prod_vector)
            else:
                cur_num_sums = self.num_classes if layer_idx == len(rg_layers)-1 else self.args.num_sums

                regions = rg_layers[layer_idx]
                for i, region in enumerate(regions):
                    product_vectors = self._region_products[region]
                    vector_name = '{}_sum_{}_{}'.format(self.name, layer_idx, i)
                    sum_vector = SumVector(product_vectors, cur_num_sums, self.args, name=vector_name)
                    self.vector_list[-1].append(sum_vector)

                    self._region_distributions[region] = sum_vector

                ps_count = ps_count + 1

        self.output_vector = self._region_distributions[self._region_graph.get_root_region()]

    def forward(self, inputs, marginalized=None):
        obj_to_tensor = {}
        for leaf_vector in self.vector_list[0]:
            obj_to_tensor[leaf_vector] = leaf_vector.forward(inputs, marginalized)

        for layer_idx in range(1, len(self.vector_list)):
            for vector in self.vector_list[layer_idx]:
                input_tensors = [obj_to_tensor[obj] for obj in vector.inputs]
                result = vector.forward(input_tensors)
                obj_to_tensor[vector] = result

        return obj_to_tensor[self.output_vector]

    @torch.no_grad()
    def reconstruct_batch(self, y_empty):
        batch_size = y_empty.shape[0]

        marginalized = torch.ones(y_empty.shape[:2]).to(device)
        self.forward(torch.zeros_like(y_empty), marginalized=marginalized)

        max_idx_tensors = {}
        for layer_idx in range(1, len(self.vector_list)):
            for vector in self.vector_list[layer_idx]:
                if isinstance(vector, SumVector):
                    max_idx_tensors[vector.name] = vector.max_child_idx

        max_idxs = max_idx_tensors
        recons = []
        for i in range(batch_size):
            recons.append(self.reconstruct(max_idxs, i, False))
        recons = torch.stack(recons, dim=0)
        # recons = np.clip(recons, 0.0, 1.0)
        return recons

    def reconstruct(self, max_idxs, case_num, sample):
        return self.output_vector.reconstruct(max_idxs, 0, case_num, sample)

    def init_params(self, init_fn):
        for layer in self.vector_list:
            for vector in layer:
                vector.init_params(init_fn)

    def num_params(self):
        result = 0
        params_per_dim = [0] * self.num_dims
        for i, layer in enumerate(self.vector_list):
            layer_result = 0
            for vector in layer:
                layer_result += vector.num_params()
                if i == 0:
                    for dim in vector.scope:
                        params_per_dim[dim] += vector.size

            print("Layer {} has {} parameters.".format(i, layer_result))
            result += layer_result
        # print(params_per_dim)
        return result

    def num_params_separately(self):
        num_sum_params = 0
        num_leaf_params = 0
        for i, layer in enumerate(self.vector_list):
            for vector in layer:
                if type(vector) == SumVector:
                    num_sum_params += vector.num_params()
                else:
                    num_leaf_params += vector.num_params()

        return num_sum_params, num_leaf_params

if __name__ == '__main__':
    rg = region_graph.RegionGraph(range(30))
    for _ in range(0, 8):
        rg.random_split(2, 2)

    args = SpnArgs()
    args.num_sums = 8
    args.num_gauss = 4
    spn = RatSpn(1, region_graph=rg, name="spn", args=args)
    spn.num_params()
    print(spn.forward(torch.rand(1, 30)))
