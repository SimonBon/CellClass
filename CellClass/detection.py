from deepcell.applications import NuclearSegmentation

# Load the image
im = imread('HeLa_nuclear.png')

# Expand image dimensions to rank 4
im = np.expand_dims(im, axis=-1)
im = np.expand_dims(im, axis=0)

# Create the application
app = NuclearSegmentation()

# create the lab
labeled_image = app.predict(image)