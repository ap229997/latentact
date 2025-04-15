import numpy as np
from scipy import linalg
import torch


def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def compute_mpjpe(pred, gt, valid=None):
    out = torch.linalg.norm(pred - gt, dim=-1)
    if valid is not None:
        if valid.sum() == 0: # otherwise 0 will be returned
            return torch.tensor(float('nan'))
        out[valid == 0] = float('nan') # set invalid values to nan,
    return torch.nanmean(out)


def compute_mpjpe_ra(pred, gt, valid=None):
    pred_ra = pred - pred[:, 0:1, :]
    gt_ra = gt - gt[:, 0:1, :]
    return compute_mpjpe(pred_ra, gt_ra, valid)


def compute_similarity_transform(S1, S2, return_transf=False): # taken from https://github.com/mkocabas/PARE/blob/master/pare/utils/eval_utils.py#L45
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    if return_transf:
        return S1_hat, (scale, R, t)
    return S1_hat


def compute_mpjpe_pa(pred, gt, valid=None):
    pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
    mpjpe_pa = []
    if valid is None:
        valid = np.ones((pred_np.shape[0]))
    for j, (c_pred, c_gt) in enumerate(zip(pred_np, gt_np)):
        if len(valid[j].shape) == 0:
            check_valid = valid[j] == 0
        else:
            check_valid = (valid[j] == 0).all()
        if np.isnan(c_gt).any() or check_valid:
            mpjpe_pa.append(np.nan)
            continue
        c_pred = c_pred.reshape(-1, 3)
        c_gt = c_gt.reshape(-1, 3)
        c_pred_hat = compute_similarity_transform(c_pred, c_gt)
        mpjpe_pa.append(np.nanmean(np.linalg.norm(c_pred_hat - c_gt, axis=1)))
    return np.nanmean(mpjpe_pa)


def compute_mpjpe_pa_first(pred, gt, valid=None): # align only the first frame
    pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy() # B, T, 21, 3
    mpjpe_pa = []
    if valid is None:
        valid = np.ones((pred_np.shape[0]))
    for j, (c_pred, c_gt) in enumerate(zip(pred_np, gt_np)):
        if np.isnan(c_gt).any() or valid[j] == 0:
            mpjpe_pa.append(np.nan)
            continue
        # compute similarity transform at the first frame
        _, transf = compute_similarity_transform(c_pred[0], c_gt[0], return_transf=True)
        scale, R, t = transf
        # apply the same transformation to all frames
        c_pred_hat = (scale * R.dot(c_pred.reshape(-1,3).T) + t).T
        c_pred_hat = c_pred_hat.reshape(-1, 21, 3)
        mpjpe_pa.append(np.nanmean(np.linalg.norm(c_pred_hat - c_gt, axis=1)))
    return np.nanmean(mpjpe_pa)


def binary_classification_metrics(logits, indices, mask=None):
    if mask is None:
        mask = torch.ones_like(indices)
    assert mask.shape[:-1] == indices.shape[:-1]
    
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    assert preds.shape == indices.shape
    preds = preds.bool()
    indices = indices.bool()

    last_dim = preds.shape[-1]
    preds = preds.reshape(-1, last_dim)
    indices = indices.reshape(-1, last_dim)
    mask = mask.reshape(-1)

    preds = preds[mask == 1]
    indices = indices[mask == 1]
    check = (preds == indices)
    
    true_positives = (check * indices).sum().item()
    false_positives = ((~indices) * preds).sum().item()
    false_negatives = (indices * (~preds)).sum().item()
    true_negatives = ((~indices) * (~preds)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    # avoid division by zero
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1