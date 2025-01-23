"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""

import numpy as np
import scipy.linalg
from utils import transformations
import quaternion
import copy

from utils.gaussian import NdimGaussian


def dz(z1: NdimGaussian, z2: NdimGaussian):  # z1 - z2
    mu1 = np.linalg.inv(z1.lam) @ z1.eta
    mu2 = np.linalg.inv(z2.lam) @ z2.eta
    Sigma1 = np.linalg.inv(z1.lam)
    Sigma2 = np.linalg.inv(z2.lam)

    dmu = mu1 - mu2
    dSigma = Sigma1
    dlambda = np.linalg.inv(dSigma)
    deta = dlambda @ dmu
    return NdimGaussian(dmu.shape[0], deta, dlambda)


def kl_divergence_gaussian(p: NdimGaussian, q: NdimGaussian) -> float:
    """
    Compute the Kullback-Leibler divergence between two NdimGaussian distributions.

    Parameters:
    p (NdimGaussian): The first Gaussian distribution.
    q (NdimGaussian): The second Gaussian distribution.

    Returns:
    float: The KL divergence between p and q.
    """
    assert p.dim == q.dim
    mu1 = p.mu
    mu2 = q.mu
    cov1 = p.sigma
    cov2 = q.sigma

    k = p.dim

    cov2_inv = np.linalg.inv(cov2)
    cov1_inv = np.linalg.inv(cov1)  
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)

    trace = np.trace(cov2_inv @ cov1)

    diff = mu2 - mu1
    quadratic_term = diff.T @ cov2_inv @ diff

    log_det_term = np.log(det_cov2 / det_cov1)

    kl = 0.5 * (trace + quadratic_term - k + log_det_term)

    if kl < 0:
        print("mu1", mu1)
        print("mu2", mu2)
        print("cov1", cov1)
        print("cov2", cov2)
        print("k", k)
        raise ValueError(f"KL divergence is negative: {kl}")

    return kl


def contraction(d_last, z1: NdimGaussian, z2: NdimGaussian) -> NdimGaussian:
    _lambda = 1.0

    # compute new lambda
    d_current = kl_divergence_gaussian(z1, z2)
    print(d_current)
    # print(d_current)
    if d_current < 1e-6:
        return z2, d_last, 0, d_current

    _dz = dz(z2, z1)

    alpha = 0.9
    d_reset = 10  # chi2*2
    d_target = alpha * d_last
    print("target ", d_target)
    if d_current <= d_target or d_current > d_reset:
        d_last = d_current
        print("already contract")
        return z2, d_current, 0, d_current

    Sigma1 = np.linalg.inv(z1.lam)
    Sigmad = np.linalg.inv(_dz.lam)
    mu1 = np.linalg.inv(z1.lam) @ z1.eta
    dmu = np.linalg.inv(_dz.lam) @ _dz.eta

    diff = np.transpose(dmu) @ np.linalg.inv(Sigma1) @ dmu
    tr = np.trace(np.linalg.inv(Sigma1) @ Sigmad)
    denom = tr + diff

    if denom < 1e-6:
        _lambda = 1
    else:
        _lambda = np.sqrt(2 * d_target / denom)

    _lambda = np.min([_lambda, 1])

    new_Sigma = Sigma1 + Sigmad * _lambda * _lambda
    new_lambda = np.linalg.inv(new_Sigma)
    new_mu = mu1 + _lambda * dmu
    new_eta = new_lambda @ new_mu
    return (
        NdimGaussian(new_mu.shape[0], new_eta, new_lambda),
        d_target,
        _lambda,
        d_current,
    )


def contract_sigma(d_last, sigma1, sigma2):
    _lambda = 1.0

    d_current = np.linalg.norm(sigma2 - sigma1)
    if d_current < 1e-6:
        return sigma2, d_last, 0, dsigma

    d_target = 0.9 * d_last
    if d_current <= d_target:
        d_last = d_current
        return sigma2, d_current, 0, d_current

    _lambda = d_target / d_current
    new_sigma = sigma1 + (sigma2 - sigma1) * _lambda
    return new_sigma, d_target, 0, d_current


class FactorGraph:
    def __init__(
        self,
        nonlinear_factors=True,
        eta_damping=0.0,
        beta=None,
        num_undamped_iters=None,
        min_linear_iters=None,
    ):

        self.var_nodes = []
        self.factors = []

        self.n_var_nodes = 0
        self.n_factor_nodes = 0
        self.n_edges = 0

        self.nonlinear_factors = nonlinear_factors

        self.eta_damping = eta_damping

        if nonlinear_factors:
            # For linearising nonlinear measurement factors.
            self.beta = beta  # Threshold change in mean of adjacent beliefs for relinearisation.
            self.num_undamped_iters = num_undamped_iters  # Number of undamped iterations after relinearisation before damping is set to 0.4
            self.min_linear_iters = min_linear_iters  # Minimum number of linear iterations before a factor is allowed to realinearise.

    def energy(self):
        """
        Computes the sum of all of the squared errors in the graph using the appropriate local loss function.
        """
        energy = 0
        for factor in self.factors:
            # Variance of Gaussian noise at each factor is weighting of each term in squared loss.
            # energy += 0.5 * np.linalg.norm(factor.compute_residual()) ** 2 / factor.adaptive_gauss_noise_var
            energy += factor.energy()
        return energy

    def compute_all_messages(self, local_relin=True):
        for factor in self.factors:
            # If relinearisation is local then damping is also set locally per factor.
            if self.nonlinear_factors and local_relin:
                if factor.iters_since_relin == self.num_undamped_iters:
                    factor.eta_damping = self.eta_damping
                factor.compute_messages(factor.eta_damping)
            else:
                factor.compute_messages(self.eta_damping)

    def update_all_beliefs(self):
        for var_node in self.var_nodes:
            var_node.update_belief()

    def compute_all_factors(self):
        for factor in self.factors:
            factor.compute_factor()

    def relinearise_factors(self):
        """
        Compute the factor distribution for all factors for which the local belief mean has deviated a distance
        greater than beta from the current linearisation point.
        Relinearisation is only allowed at a maximum frequency of once every min_linear_iters iterations.
        """
        if self.nonlinear_factors:
            for factor in self.factors:
                adj_belief_means = np.array([])
                for belief in factor.adj_beliefs:
                    adj_belief_means = np.concatenate(
                        (adj_belief_means, np.linalg.inv(belief.lam) @ belief.eta)
                    )
                if (
                    np.linalg.norm(factor.linpoint - adj_belief_means) > self.beta
                    and factor.iters_since_relin >= self.min_linear_iters
                ):
                    factor.compute_factor(linpoint=adj_belief_means)
                    factor.iters_since_relin = 0
                    factor.eta_damping = 0.0
                else:
                    factor.iters_since_relin += 1

    def robustify_all_factors(self):
        for factor in self.factors:
            factor.robustify_loss()

    def synchronous_iteration(self, local_relin=True, robustify=False):
        if robustify:
            self.robustify_all_factors()
        if self.nonlinear_factors and local_relin:
            self.relinearise_factors()
        self.compute_all_messages(local_relin=local_relin)
        self.update_all_beliefs()

    def joint_distribution_inf(self):
        """
        Get the joint distribution over all variables in the information form
        If nonlinear factors, it is taken at the current linearisation point.
        """

        eta = np.array([])
        lam = np.array([])
        var_ix = np.zeros(len(self.var_nodes)).astype(int)
        tot_n_vars = 0
        for var_node in self.var_nodes:
            var_ix[var_node.variableID] = int(tot_n_vars)
            tot_n_vars += var_node.dofs
            eta = np.concatenate((eta, var_node.prior.eta))
            if var_node.variableID == 0:
                lam = var_node.prior.lam
            else:
                lam = scipy.linalg.block_diag(lam, var_node.prior.lam)

        for factor in self.factors:
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                vID = adj_var_node.variableID
                # Diagonal contribution of factor
                eta[var_ix[vID] : var_ix[vID] + adj_var_node.dofs] += factor.factor.eta[
                    factor_ix : factor_ix + adj_var_node.dofs
                ]
                lam[
                    var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                    var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                ] += factor.factor.lam[
                    factor_ix : factor_ix + adj_var_node.dofs,
                    factor_ix : factor_ix + adj_var_node.dofs,
                ]
                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if other_adj_var_node.variableID > adj_var_node.variableID:
                        other_vID = other_adj_var_node.variableID
                        # Off diagonal contributions of factor
                        lam[
                            var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                            var_ix[other_vID] : var_ix[other_vID]
                            + other_adj_var_node.dofs,
                        ] += factor.factor.lam[
                            factor_ix : factor_ix + adj_var_node.dofs,
                            other_factor_ix : other_factor_ix + other_adj_var_node.dofs,
                        ]
                        lam[
                            var_ix[other_vID] : var_ix[other_vID]
                            + other_adj_var_node.dofs,
                            var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                        ] += factor.factor.lam[
                            other_factor_ix : other_factor_ix + other_adj_var_node.dofs,
                            factor_ix : factor_ix + adj_var_node.dofs,
                        ]
                    other_factor_ix += other_adj_var_node.dofs
                factor_ix += adj_var_node.dofs

        return eta, lam

    def joint_distribution_cov(self):
        """
        Get the joint distribution over all variables in the covariance.
        If nonlinear factors, it is taken at the current linearisation point.
        """
        eta, lam = self.joint_distribution_inf()
        sigma = np.linalg.inv(lam)
        mu = sigma @ eta
        return mu, sigma

    def get_means(self):
        """
        Get an array containing all current estimates of belief means.
        """
        mus = np.array([])
        for var_node in self.var_nodes:
            mus = np.concatenate((mus, var_node.mu))
        return mus

    def save_to_g2o(self, filename):
        """
        Save the factor graph to a g2o file.
        """
        print("saving to g2o: ", filename)
        with open(filename, "w") as file:
            for factor in self.factors:
                file.write(
                    f"EDGE_SE3:QUAT {factor.adj_var_nodes[0].variableID} {factor.adj_var_nodes[1].variableID} "
                )
                measurement_vec = factor.measurement
                # convert to quaternion
                q = quaternion.from_rotation_vector(measurement_vec[3:])
                # save xyz and quaternion
                file.write(" ".join([str(x) for x in measurement_vec[:3]]) + " ")
                file.write(" ".join([str(q.x), str(q.y), str(q.z), str(q.w)]) + " ")
                # expand factor.gauss_noise_var to matrix
                sigma = np.diag(1.0 / (factor.gauss_noise_var))
                info = np.linalg.inv(sigma)
                file.write(
                    " ".join([str(x) for x in info[np.triu_indices(6)].flatten()])
                    + "\n"
                )
            print(f"saving {len(self.var_nodes)} nodes")
            for var_node in self.var_nodes:
                file.write(f"VERTEX_SE3:QUAT {var_node.variableID} ")
                q = quaternion.from_rotation_vector(var_node.mu[3:])
                file.write(" ".join([str(x) for x in var_node.mu[:3]]) + " ")
                file.write(" ".join([str(x) for x in q.components]) + "\n")


class VariableNode:
    def __init__(self, variable_id, dofs, **kwargs):

        self.variableID = variable_id
        self.adj_factors = []

        # Node variables are position of landmark in world frame. Initialize variable nodes at origin
        self.mu = np.zeros(dofs)
        self.Sigma = np.zeros([dofs, dofs])

        self.belief = NdimGaussian(dofs)
        self.belief.eta = np.zeros(dofs)
        self.belief.lam = np.eye(dofs)

        self.prior = NdimGaussian(dofs)
        self.prior_lambda_end = (
            -1
        )  # -1 flag if the sigma of self.prior is prior_sigma_end
        self.prior_lambda_logdiff = -1

        self.dofs = dofs
        self.d_last = np.inf
        self.d_sigma = np.inf
        self.contraction = kwargs.get("contraction", True)

    def update_belief(self):
        """
        Update local belief estimate by taking product of all incoming messages along all edges.
        Then send belief to adjacent factor nodes.
        """
        # Update local belief
        # eta=self.prior.eta.copy()
        # lam=self.prior.lam.copy()
        eta = np.zeros(self.dofs)
        lam = np.eye(self.dofs) 
        for i, factor in enumerate(self.adj_factors):
            message_ix = factor.adj_vIDs.index(self.variableID)
            eta_inward, lam_inward = (
                factor.messages[message_ix].eta,
                factor.messages[message_ix].lam,
            )
            eta += eta_inward
            lam += lam_inward

            if np.any(np.diag(lam_inward) < 0):
                print(f"variable {self.variableID} lam_inward: {lam_inward}")
                raise ValueError("lam_inward has negative diagonal elements")
                sys.exit(1)

        # if np.any(np.linalg.eigvals(lam) < 0):
        #     print(f"variable {self.variableID} new_belief.lam: {lam}")
        #     raise ValueError("message lam not invertable")
        #     sys.exit(1)

        new_belief = NdimGaussian(self.dofs, eta, lam)

        # Send belief to adjacent factors
        for factor in self.adj_factors:
            belief_ix = factor.adj_vIDs.index(self.variableID)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = (
                new_belief.eta,
                new_belief.lam,
            )
        if self.contraction:
            # check if current belief is valid by checking if self.belief.lam invertable
            old_belief = None
            if np.linalg.det(self.belief.lam) == 0:
                old_belief = NdimGaussian(
                    self.dofs, np.zeros(self.dofs), np.eye(self.dofs)
                )
            else:
                old_belief = copy.deepcopy(self.belief)

            # new_sigma, self.d_sigma, _lambda, d_current = contract_sigma(
            #     self.d_sigma,
            #     np.linalg.inv(old_belief.lam),
            #     np.linalg.inv(new_belief.lam),
            # )
            # new_belief.lam = np.linalg.inv(new_sigma)
            new_belief, self.d_last, _lambda, d_current = contraction(
                self.d_last, old_belief, new_belief
            )
            lam_norm = np.linalg.norm(old_belief.lam)
            print(
                f"variable {self.variableID} lambda: {_lambda}, d_current: {d_current}, d_target: {self.d_last}, lam: {lam_norm}"
            )

        if np.any(np.linalg.eigvals(new_belief.lam) < 0):
            print(f"variable {self.variableID} new_belief.lam: {new_belief.lam}")
            raise ValueError("new_belief.lam has negative diagonal elements")
            sys.exit(1)

        self.belief.eta = new_belief.eta.copy()
        self.belief.lam = new_belief.lam.copy()
        self.Sigma = np.linalg.inv(self.belief.lam)
        self.mu = self.Sigma @ self.belief.eta




class ConstantVariableNode(VariableNode):
    # inherit init from VariableNode
    def __init__(self, variable_id, dofs):
        super().__init__(variable_id, dofs)

    # override update_belief method to be empty
    def update_belief(self):
        for factor in self.adj_factors:
            belief_ix = factor.adj_vIDs.index(self.variableID)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = (
                self.prior.eta,
                self.prior.lam,
            )

        self.belief.eta = self.prior.eta.copy()
        self.belief.lam = self.prior.lam.copy()


class Factor:
    def __init__(
        self,
        factor_id,
        adj_var_nodes,
        measurement,
        gauss_noise_std,
        meas_fn,
        jac_fn,
        loss=None,
        mahalanobis_threshold=2,
        *args,
    ):
        """
        n_stds: number of standard deviations from mean at which loss transitions to robust loss function.
        """

        self.factorID = factor_id

        self.dofs_conditional_vars = 0
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = []
        self.adj_beliefs = []
        self.messages = []

        for adj_var_node in self.adj_var_nodes:
            self.dofs_conditional_vars += adj_var_node.dofs
            self.adj_vIDs.append(adj_var_node.variableID)
            self.adj_beliefs.append(NdimGaussian(adj_var_node.dofs))
            self.messages.append(NdimGaussian(adj_var_node.dofs))

        self.factor = NdimGaussian(self.dofs_conditional_vars)
        self.linpoint = np.zeros(self.dofs_conditional_vars)  # linearisation point

        self.measurement = measurement

        # Measurement model
        self.gauss_noise_var = gauss_noise_std**2
        self.meas_fn = meas_fn
        self.jac_fn = jac_fn
        self.args = args

        # Robust loss function
        self.adaptive_gauss_noise_var = gauss_noise_std**2
        self.loss = loss
        self.mahalanobis_threshold = mahalanobis_threshold
        self.robust_flag = False

        # Local relinearisation
        self.eta_damping = 0.0
        self.iters_since_relin = 1

    def compute_residual(self):
        """
        Calculate the reprojection error vector.
        """
        adj_belief_means = []
        for belief in self.adj_beliefs:
            adj_belief_means = np.concatenate(
                (adj_belief_means, np.linalg.inv(belief.lam) @ belief.eta)
            )
            assert np.all(np.linalg.eigvals(belief.lam) > 0)
        d = self.meas_fn(adj_belief_means, *self.args) - self.measurement
        return d

    def energy(self):
        """
        Computes the squared error using the appropriate loss function.
        """
        if isinstance(self.adaptive_gauss_noise_var, float):
            return (
                0.5
                * np.linalg.norm(self.compute_residual()) ** 2
                / self.adaptive_gauss_noise_var
            )
        else:
            r = self.compute_residual()
            # self.adaptive_gauss_noise_var is a vector of variances for each dimension of the residual
            assert r.shape[0] == len(self.adaptive_gauss_noise_var)
            info = np.diag(1 / self.adaptive_gauss_noise_var)
            return 0.5 * r.T @ np.diag(1 / self.adaptive_gauss_noise_var) @ r

    def compute_factor(self, linpoint=None, update_self=True):
        """
        Compute the factor given the linearisation point.
        If not given then linearisation point is mean of belief of adjacent nodes.
        If measurement model is linear then factor will always be the same regardless of linearisation point.
        """
        if linpoint is None:
            self.linpoint = []
            for belief in self.adj_beliefs:
                self.linpoint += list(np.linalg.inv(belief.lam) @ belief.eta)
        else:
            self.linpoint = linpoint

        J = self.jac_fn(self.linpoint, *self.args)
        pred_measurement = self.meas_fn(self.linpoint, *self.args)

        meas_model_lambda = (
            np.eye(len(self.measurement)) / self.adaptive_gauss_noise_var
        )
        lambda_factor = J.T @ meas_model_lambda @ J
        eta_factor = (J.T @ meas_model_lambda) @ (
            J @ self.linpoint + self.measurement - pred_measurement
        )

        if update_self:
            self.factor.eta, self.factor.lam = eta_factor, lambda_factor

        return eta_factor, lambda_factor

    def robustify_loss(self):
        """
        Rescale the variance of the noise in the Gaussian measurement model if necessary and update the factor
        correspondingly.
        """
        old_adaptive_gauss_noise_var = self.adaptive_gauss_noise_var
        if self.loss is None:
            self.adaptive_gauss_noise_var = self.gauss_noise_var

        else:
            adj_belief_means = np.array([])
            for belief in self.adj_beliefs:
                adj_belief_means = np.concatenate(
                    (adj_belief_means, np.linalg.inv(belief.lam) @ belief.eta)
                )
            pred_measurement = self.meas_fn(self.linpoint, *self.args)

            if (
                self.loss == "huber"
            ):  # Loss is linear after Nstds from mean of measurement model
                mahalanobis_dist = np.linalg.norm(
                    self.measurement - pred_measurement
                ) / np.sqrt(self.gauss_noise_var)
                if mahalanobis_dist > self.mahalanobis_threshold:
                    self.adaptive_gauss_noise_var = (
                        self.gauss_noise_var
                        * mahalanobis_dist**2
                        / (
                            2
                            * (
                                self.mahalanobis_threshold * mahalanobis_dist
                                - 0.5 * self.mahalanobis_threshold**2
                            )
                        )
                    )
                    self.robust_flag = True
                else:
                    self.robust_flag = False
                    self.adaptive_gauss_noise_var = self.gauss_noise_var

            elif (
                self.loss == "constant"
            ):  # Loss is constant after Nstds from mean of measurement model
                mahalanobis_dist = np.linalg.norm(
                    self.measurement - pred_measurement
                ) / np.sqrt(self.gauss_noise_var)
                if mahalanobis_dist > self.mahalanobis_threshold:
                    self.adaptive_gauss_noise_var = mahalanobis_dist**2
                    self.robust_flag = True
                else:
                    self.robust_flag = False
                    self.adaptive_gauss_noise_var = self.gauss_noise_var

        scale_factor = old_adaptive_gauss_noise_var / self.adaptive_gauss_noise_var
        if not isinstance(scale_factor, float):
            assert (
                len(scale_factor) == len(self.factor.eta) // 2
            )  # If scale_factor is half the size of eta
            scale_factor = np.hstack(
                [scale_factor, scale_factor]
            )  # Duplicate for both poses
        self.factor.eta *= scale_factor
        self.factor.lam *= scale_factor

    def compute_messages(self, eta_damping):
        """
        Compute all outgoing messages from the factor.
        """
        messages_eta, messages_lam = [], []
        start_dim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = self.factor.eta.copy(), self.factor.lam.copy()

            # Take product of factor with incoming messages
            mess_start_dim = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_factor[mess_start_dim : mess_start_dim + var_dofs] += (
                        self.adj_beliefs[var].eta - self.messages[var].eta
                    )
                    lam_factor[
                        mess_start_dim : mess_start_dim + var_dofs,
                        mess_start_dim : mess_start_dim + var_dofs,
                    ] += (
                        self.adj_beliefs[var].lam - self.messages[var].lam
                    )
                mess_start_dim += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            mess_dofs = self.adj_var_nodes[v].dofs
            eo = eta_factor[start_dim : start_dim + mess_dofs]
            eno = np.concatenate(
                (eta_factor[:start_dim], eta_factor[start_dim + mess_dofs :])
            )

            loo = lam_factor[
                start_dim : start_dim + mess_dofs, start_dim : start_dim + mess_dofs
            ]
            lono = np.hstack(
                (
                    lam_factor[start_dim : start_dim + mess_dofs, :start_dim],
                    lam_factor[
                        start_dim : start_dim + mess_dofs, start_dim + mess_dofs :
                    ],
                )
            )
            lnoo = np.vstack(
                (
                    lam_factor[:start_dim, start_dim : start_dim + mess_dofs],
                    lam_factor[
                        start_dim + mess_dofs :, start_dim : start_dim + mess_dofs
                    ],
                )
            )
            lnono = np.block(
                [
                    [
                        lam_factor[:start_dim, :start_dim],
                        lam_factor[:start_dim, start_dim + mess_dofs :],
                    ],
                    [
                        lam_factor[start_dim + mess_dofs :, :start_dim],
                        lam_factor[start_dim + mess_dofs :, start_dim + mess_dofs :],
                    ],
                ]
            )

            # Compute outgoing messages
            messages_lam.append(loo - lono @ np.linalg.inv(lnono) @ lnoo)
            new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno
            messages_eta.append(
                (1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta
            )
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]

            if np.any(np.linalg.eigvals(self.messages[v].lam) < 0):
                print(f"factor {self.factorID} message {v} lam: {self.messages[v].lam}")

                for belief in self.adj_beliefs:
                    print(f"belief {belief.lam}")

                raise ValueError("computed lam not invertable")
                sys.exit(1)
