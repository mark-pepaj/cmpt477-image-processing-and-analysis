
"""
Given an image like this, we can split the image
into slices where elements in a slice are 1 if
the image has an intensity greater than or equal
to their slice index.is a 1 and 0 otherwise.

Example is the intensity 10 at x=0, y = 0.
Lets say there are 15 max intensities in an image.
If we cut the image into slices where each slice corresponds to an intensity
then we get 15 slices.
For slice 1 to 10 there will be a 1 in the x=0, y=0 index.
When we add the slices up we get a 10 in x=0, y=0.

This is the idea behind boolean threshold filtering.
Now we can analyze the slices.

If a window in a slice has a 1 in the center and there are more ones
than zeros in the neighborhood, the center stays as 1.
But if there is a 1 in the center and there are more zeros than ones
in the neighborhood, then the center becomes 0

Vice versa if there is a 0 in the center pixel.

0 1 0      0 1 0
1 0 1  ->  1 1 1
1 1 1      1 1 1


img = np.array([[0, 0, 0, 0,       0,     0,   0,   0, 0],
                [0, 0, 0, 0,       150,   0,   0,   0, 0],
                [0, 0, 0,     150, 200, 150,   0,   0, 0],
                [0, 0,   150, 200, 255, 200, 150,   0, 0],
                [0, 150, 200, 255, 255, 255, 200, 150, 0],
                [0, 0,   150, 200, 255, 200, 150,   0, 0],
                [0, 0, 0,     150, 200, 150,   0,   0, 0],
                [0, 0, 0, 0,       150,   0,   0,   0, 0],
                [0, 0, 0, 0,       0,     0,   0,   0, 0]]).astype(np.uint8)
"""

