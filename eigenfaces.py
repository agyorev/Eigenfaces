import os
import cv2
import sys
import random
import numpy as np

"""
Example:
    $> python2.7 eigenfaces.py att_faces
"""

class Eigenfaces(object):
    faces_count = 40

    faces_dir = '.'                 # directory path to the AT&T faces

    l = 6 * faces_count             # training images count
    m = 92                          # number of columns of the image
    n = 112                         # number of rows of the image
    mn = m * n                      # length of the column vector

    def __init__(self, _faces_dir = '.'):
        self.faces_dir = _faces_dir
        self.training_ids = []

        L = np.empty(shape=(self.mn, self.l))

        cur_img = 0
        for face_id in xrange(1, self.faces_count + 1):

            training_ids = random.sample(range(1, 10), 6) # the id's of the 6 random training images
            self.training_ids.append(training_ids)        # remembering the training id's for later

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir, 's' + str(face_id), str(training_id) + '.pgm')
                print '> reading file: ' + path_to_img

                img = cv2.imread(path_to_img, 0)          # read a grayscale image
                img_col = np.array(img).flatten()         # flatten the 2d image into 1d

                L[:, cur_img] = img_col[:]                # set the cur_img-th column to the current training image
                cur_img += 1

        self.mean_img_col = np.sum(L, axis=1) / self.l    # get the mean of all images / over the rows of L

        for j in xrange(0, self.l):                       # subtract from all training images
            L[:, j] -= self.mean_img_col[:]

        # instead of computing the covariance matrix as
        # L*L^T, we set C = L^T*L, and end up with way
        # smaller and computentionally inexpensive one
        C = np.matrix(L.transpose()) * np.matrix(L)

        U, s, V = np.linalg.svd(C, full_matrices=True)    # compute the SVD decomposition

        # L * C * v_i       = lam_i * L * v_i
        # L * L^T * L * v_i = lam_i * L * v_i
        # => e-vectors and e-values of L * L^T
        #    are L * v_i and lam_i respectively
        self.evectors = L * V
        norms = np.linalg.norm(self.evectors, axis=0)
        self.evectors = self.evectors / norms

        # include only the first k evectors/values so
        # that they include approx. 85% of the energy
        evalues_sum = sum(s[:])
        evalues_count = 0
        evalues_energy = 0.0
        for evalue in s:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= 0.85:
                break

        self.evalues = s[0:evalues_count]
        self.evectors = self.evectors[:, 0:evalues_count]

        # energy_count rows for the l columns/training images
        self.W = self.evectors.transpose() * L            # computing the weights

    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                  # read as a grayscale image
        img_col = np.array(img).flatten()                 # flatten the image
        img_col -= self.mean_img_col                      # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))       # from row vector to col vector

        # projecting the normalized probe onto the
        # Eigenspace, to find out the weights
        S = self.evectors.transpose() * img_col

        # finding the min ||W_j - S||
        diff = self.W - S
        norms = np.linalg.norm(diff, axis=0)
        print norms, norms.shape

        closest_face_id = np.argmin(norms)                # the id [0..240) of the minerror face to the sample
        print closest_face_id

if __name__ == "__main__":
    print Eigenfaces.faces_dir
    efaces = Eigenfaces(str(sys.argv[1]))
    Eigenfaces.classify(efaces, 'att_faces/s1/1.pgm')

#mean_img = np.reshape(mean_img_col, (self.n, self.m))
#img2 = mean_img
#cv2.imwrite('test.png', img2)

