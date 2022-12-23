## function to evaluate: accuracy, precision, recall, f1_score

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_accuracy(y_true, y_pred):
    """
    Calculate model accuracy, precision, recall and f1_score for a binary classification model
    """
    # calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # calculate model precision, recall and f1_score using 'weighted average'
    model_precision, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
    ## model results
    model_results = {"Accuracy Score": model_accuracy,
                     "Precision Score" : model_precision,
                     "Recall Score": model_recall,
                     "F1 Score": model_f1_score}

    ## returing results
    return model_results        
  
    
    
import tensorflow as tf

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img
  
