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
Normal AT&T face data set training and recognition:

    $> python2.7 eigenfaces.py att_faces
    
or if we want to include also the celebrity faces evaluation:

    $> python2.7 eigenfaces.py att_faces celebrity_faces

Results
---
Under the `results/` folder there will be a `att_results.txt` file containing detailed results from the evaluation over the test images (40% of all faces).

If a celebrity data set was specified, for each face in the celebrity data set, there will be a folder with results for it, including the Top 5 matches from the AT&T faces, as well as the similarity score between them.

Plotting
---
We can also plot (using `gnuplot`) the accuracy results, depending on how much energy we want to use to recognise the faces. Currently the different energy values to be tested are hard-coded to be multiples of 5, but this can easily be changed form `energy.py`.

    $> python2.7 energy.py att_faces
    $> gnuplot plot_energy.gpi

Algorithm Reference
---
[Link](http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#algorithmic-description) to the description of the algorithm in the OpenCV documentation.
