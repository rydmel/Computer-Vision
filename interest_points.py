# Ryan D'Mello
#CMSC 25040
#HW2

import math
import random
import numpy as np
from canny import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 4.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points

   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""


def find_interest_points(image, max_points=100, scale=1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   # Hession matrix calculation
   dx, dy = sobel_gradients(image)
   Ixx = np.square(dx)
   Ixy = np.multiply(dx, dy)
   Iyy = np.square(dy)

   # Convolution
   Hx2 = conv_2d_gaussian(Ixx, scale)
   Hxy = conv_2d_gaussian(Ixy, scale)
   Hy2 = conv_2d_gaussian(Iyy, scale)

   # Harris Response
   alpha = 0.08
   Tr = Hx2 + Hy2
   harris = np.multiply(Hx2, Hy2) - np.square(Hxy) - alpha * np.square(Tr)
   theta = np.arctan2(dy, dx)
   response = nonmax_suppress(harris, theta)  # nonmax suppression

   # sort array with size max_points
   dim_x, dim_y = np.shape(response)
   M = []
   for i in range(dim_x):
      for j in range(dim_y):
         M.append((i, j, response[i, j]))
   M.sort(key=lambda x: x[2], reverse=True)
   M = M[0:max_points]

   # make final np arrays
   xs = np.array([p[0] for p in M])
   ys = np.array([p[1] for p in M])
   scores = np.array([p[2] for p in M])
   ##########################################################################
   return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################

   image = mirror_border(image, wx=4, wy=4)
   n = len(xs)
   dx, dy = sobel_gradients(image)
   theta, L = get_theta(dx, dy)
   feats = np.zeros([n, 72])

   for idx, x in enumerate(xs):
      ctr = 0
      wx = [-3, 0, 3]
      wy = [-3, 0, 3]
      x += 4
      y = ys[idx] + 4
      for i in wx:
         for j in wy:
            update_feats(idx, x, y, i, j, L, theta, feats, ctr)

   ##########################################################################
   return feats


def update_feats(idx, x, y, i, j, L, theta, feats, ctr):
   xw = math.floor(x + i)
   yw = math.floor(y + j)
   theta_new = theta[xw - 1:xw + 1, yw - 1:yw + 1]
   window = np.empty(8)
   for idx_row, t_row in enumerate(theta_new):
      for idx_col, t_val in enumerate(t_row):
         if (t_val + math.pi)//(math.pi/4) != 8:
            z = int((t_val + math.pi)/(math.pi/4))
         else:
            z = 0
         window[z] += 1
   for num in range(8):
      feats[idx, 8*ctr + i] = window[num]
   ctr +=1


def get_theta(dx, dy):
   t = np.arctan2(dy, dx)
   L = np.sqrt((dx ** 2) + (dy ** 2))
   return t, L

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion. Note that you are required to implement the naive
   linear NN search. For 'lsh' and 'kdtree' search mode, you could do either to
   get full credits.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices. You are required to report the efficiency comparison
   between different modes by measure the runtime (check the benchmarking related
   codes in hw2_example.py).

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())
      mode     - 'naive': performs a brute force NN search

               - 'lsh': Implementing the local senstive hashing (LSH) approach
                  for fast feature matching. In LSH, the high dimensional
                  feature vectors are randomly projected into low dimension
                  space which are further binarized as boolean hashcodes. As we
                  group feature vectors by hashcodes, similar vectors may end up
                  with same 'bucket' with high propabiltiy. So that we can
                  accelerate our nearest neighbour matching through hierarchy
                  searching: first search hashcode and then find best
                  matches within the bucket.
                  Advice for impl.:
                  (1) Construct a LSH class with method like
                  compute_hash_code   (handy subroutine to project feature
                                      vector and binarize)
                  generate_hash_table (constructing hash table for all input
                                      features)
                  search_hash_table   (handy subroutine to search hash table)
                  search_feat_nn      (search nearest neighbour for input
                                       feature vector)
                  (2) It's recommended to use dictionary to maintain hashcode
                  and the associated feature vectors.
                  (3) When there is no matching for queried hashcode, find the
                  nearest hashcode as matching. When there are multiple vectors
                  with same hashcode, find the cloest one based on original
                  feature similarity.
                  (4) To improve the robustness, you can construct multiple hash tables
                  with different random project matrices and find the closest one
                  among all matched queries.
                  (5) It's recommended to fix the random seed by random.seed(0)
                  or np.random.seed(0) to make the matching behave consistenly
                  across each running.

               - 'kdtree': construct a kd-tree which will be searched in a more
                  efficient way. https://en.wikipedia.org/wiki/K-d_tree
                  Advice for impl.:
                  (1) The most important concept is to construct a KDNode. kdtree
                  is represented by its root KDNode and every node represents its
                  subtree.
                  (2) Construct a KDNode class with Variables like data (to
                  store feature points), left (reference to left node), right
                  (reference of right node) index (reference of index at original
                  point sets)and Methods like search_knn.
                  In search_knn function, you may specify a distance function,
                  input two points and returning a distance value. Distance
                  values can be any comparable type.
                  (3) You may need a user-level create function which recursively
                  creates a tree from a set of feature points. You may need specify
                  a axis on which the root-node should split to left sub-tree and
                  right sub-tree.


   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""

def match_features(feats0, feats1, scores0, scores1, mode='naive'):
   if mode != 'naive':
      matches, scores = nearest_neighbor_KD(feats0, feats1)
   else:
      matches, scores = naive_linear_search(feats0, feats1)
   return matches, scores


class KDnode():
   def __init__(self, data=None, indices=None, left=None, right=None):
      self.data = data
      self.indices = indices
      self.left = left
      self.right = right

def naive_linear_search(feats0, feats1):
   N0, K0 = np.shape(feats0)
   matches = np.zeros(N0)
   scores = np.zeros(N0)
   for i in range(N0):
      matches[i], scores[i] = nearest_neighbor_feature(feats0[i], feats1)
   return matches, scores


def create_KDtree(feats, indices, mid, ind_splits, maxdepth=7):
   kdtree = KDnode(feats, indices)
   left_indices = []
   right_indices = []

   if maxdepth == 0:
      return None

   for i in indices:
      if (mid[ind_splits[-maxdepth]] <= feats[i][ind_splits[-maxdepth]]):
         right_indices.append(i)
      else:
         left_indices.append(i)
   if right_indices == []:
      kdtree.right = None
   else:
      kdtree.right = create_KDtree(feats, right_indices, mid, ind_splits, maxdepth=maxdepth - 1)
   if left_indices == []:
      kdtree.left = None
   else:
      kdtree.left = create_KDtree(feats, right_indices, mid, ind_splits, maxdepth=maxdepth - 1)
   return kdtree


def nearest_neighbor_KD(feats0, feats1):
   N0, K0 = np.shape(feats0)
   match_arr = np.zeros(N0, dtype=int)
   scores = np.zeros(N0)
   N1, K1 = np.shape(feats1)
   indices = [i for i in range(N1)]
   splits = random.sample(range(0, K1), 7)
   mid = np.median(feats1, axis=0)
   next_kdtree = create_KDtree(feats1, indices, mid, splits)

   for i in range(N0):
      next_kdtree_copy = next_kdtree
      feat_indices = next_kdtree.indices
      for split in splits:
         if next_kdtree_copy:
            feat_indices = next_kdtree_copy.indices
            if feats0[i][split] < mid[split]:
               next_kdtree_copy = next_kdtree_copy.left
            else:
               next_kdtree_copy = next_kdtree_copy.right
         else:
            break
      match_arr[i], scores[i] = nearest_neighbor_feature(feats0[i], feats1[feat_indices])

   return match_arr, scores


def nearest_neighbor_feature(feat0, feats1):
   N0, K0 = np.shape(feats1)
   dist1_min = 10**10
   dist2_min = 10**10
   i_min = 0
   for i in range(N0):
      distance = np.linalg.norm(feat0 - feats1[i])
      if distance < dist1_min:
         dist2_min = dist1_min
         i_min = i
         dist1_min = distance
      elif distance < dist2_min:
         dist2_min = distance
   if (dist2_min != 0):
      score = dist1_min/dist2_min
   else:
      score = 1
   return i_min, score

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################

   x_translate = []
   y_translate = []

   for idx_x in range(matches.shape[0]):
      idx_y = matches[idx_x]
      dx = xs0[idx_x] - xs1[idx_y]
      dy = ys0[idx_x] - ys1[idx_y]
      x_translate.append(dx)
      y_translate.append(dy)

   x_min = min(x_translate)
   x_max = max(x_translate)
   y_min = min(y_translate)
   y_max = max(y_translate)
   votes = np.zeros((max(x_translate) - min(x_translate) + 1, max(y_translate) - min(y_translate) + 1))

   for num in range(len(x_translate)):
      votes[x_translate[num] - x_min, y_translate[num] - y_min] += scores[num]

   max_score = 0
   for i, row in np.ndenumerate(votes):
      for j, k in np.ndenumerate(row):
         if k > max_score:
            max_score = k
            tx = i  + x_min
            ty = j + y_min
         else:
            pass
   ##########################################################################
   return tx, ty, votes
