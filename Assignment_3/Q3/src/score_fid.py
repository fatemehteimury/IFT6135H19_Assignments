import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np

SVHN_PATH = "../SVHN_dataset"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator, testset_feature_iterator, sample_len, testset_len):

    # fid_score = || mu_s - mu_t ||^2 + Trace(sigma_s + sigma_p - 2(sigma_s*sigma_p)^(1/2)) 
    # make sure that statistics are calculated for same number of examples
    # otherwise it will be biased towards one or other

    exam_len = min(sample_len, testset_len)

    # get features for all examples 
    all_sample_features = np.zeros((exam_len, 512)) 
    all_testset_features = np.zeros((exam_len, 512)) 
    
    for i in range(exam_len):
        all_sample_features[i,:] = next(sample_feature_iterator)

    for i in range(exam_len):
        all_testset_features[i,:] = next(testset_feature_iterator)
 
    print(all_sample_features.min())
    print(all_sample_features.max())
    print(all_testset_features.min())
    print(all_testset_features.max())
    # calculate mu and sigma for samples and testset
    mu_s = np.mean(all_sample_features, axis=0) 
    mu_t = np.mean(all_testset_features, axis=0) 
    
    print("mu_s:", mu_s[0:10])
    print("mu_t:", mu_t[0:10])

    print("mu_s - mu_t:", (mu_s - mu_t)[0:10])
    
    sigma_s = np.cov(all_sample_features, rowvar=False)
    sigma_t = np.cov(all_testset_features, rowvar=False)

    # calculate difference between sample mean and test mean
    # || mu_s - mu_t ||^2
    mu_diff = mu_s - mu_t
    mu = mu_diff.dot(mu_diff)
    print("mu: ", mu)

    # calculate trace of sample sigma and test sigma
    trace_s = np.trace(sigma_s)
    trace_t = np.trace(sigma_t)
    print("trace_s: ", trace_s)
    print("trace_t: ", trace_t)

    # calculate trace of sqrt of sigma product
    # ref: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py#L411
    #
    # Trace(sqrt(sigma_s sigma_t)) = sum(eigenvalues(sqrt(sigma_s sigma_t)))
    #                              = sum(sqrt(eigenvalues(A A B B)))       where, A = sqrt(sigma_s) and B = sqrt(sigma_t)
    #                              = sum(sqrt(eigenvalues(A B B A)))
    #                              = sum(eigenvalues(sqrt(A B B A)))
    #                              = Trace(sqrt(A B B A))
    #                              = Trace(sqrt(A sigma_t A))
    # 
    # A  = sqrt(sigma_s)
    #    = sqrt(U*diag(S)*Vh)
    #    = U * diag(sqrt(S)) * Vh

    u, s, vh = np.linalg.svd(sigma_s, full_matrices=True)
    si = np.where(np.less(s, 1e-10), s, np.sqrt(s))
    sqrt_sigma_s =  np.matmul(u, np.matmul(np.diag(si), vh))

    sqrt_sigma_s_sigma_t_sqrt_sigma_s = np.matmul(sqrt_sigma_s, np.matmul(sigma_t, sqrt_sigma_s)) 
    
    u, s, vh = np.linalg.svd(sqrt_sigma_s_sigma_t_sqrt_sigma_s, full_matrices=True)
    si = np.where(np.less(s, 1e-10), s, np.sqrt(s))
    sqrt_sigma_s_sigma_t =  np.matmul(u, np.matmul(np.diag(si), vh))

    trace_p = np.trace(sqrt_sigma_s_sigma_t)
    print("trace_p: ", trace_p)

    fid_score = mu + trace_s + trace_t - 2*trace_p

    return fid_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory, PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f, len(sample_loader.dataset), len(test_loader.dataset))
    print("FID score:", fid_score)
