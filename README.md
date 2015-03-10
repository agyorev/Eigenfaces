AT&T Eigenfaces
===
A Python class that implements the Eigenfaces algorithm
for face recognition, using eigen decomposition and
principle component analysis.

We use the AT&T data set, with 60% of the images as train
and the rest 40% as a test set, including 85% of the 
overall energy, in order to reduce the number of computations.

Additionally, we use a small set of celebrity images to
find the best AT&T matches to them.

All images should have the same size, namely (92 width, 112 height).


Example Calls
---

    $> python2.7 eigenfaces.py att_faces celebrity_faces
    $> python2.7 eigenfaces.py att_faces

Results
---
Under the `results/` folder there will be a `results.txt` file containing detailed results from the evaluation over the test images (40% of all faces).

If a celebrity data set was specified, for each face in the celebrity data set, there will be a folder with results for it, including the Top 5 matches from the AT&T faces, as well as the similarity score between them.

Algorithm Reference
---
[Link](http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#algorithmic-description) to the description of the algorithm in the OpenCV documentation.
