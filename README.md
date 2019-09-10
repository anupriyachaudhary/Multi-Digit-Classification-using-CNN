## Multi Digit Classification using CNN
### Preprocessing
- Download a dataset of single well cropped and full house numbers from SVHN dataset and preprocess the images.
- Additionally, preprocess the full house images to extract negative images which have contextual information like corners, blank areas and digit edges. This step is important to create a robust training data which will minimize false positives when working with real images.

### Classification
- Train three different CNN models: self-designed architecture, VGG-16, and Pre-Trained VGG-16 on the SVHN single cropped house numbers.
- Save the weights from the single digit classifier.
- Check Testing and training accuracies and choose the best model for detection phase.
Improvement- Create a multi-digit classifier as opposed to single-digit classifier to replicate work of Google. [Goodfellow]

### Detection
- Use Fixed size sliding windows, which slide left to right and top to bottom to localize the digits.
- Use image pyramid technique to make the detection model size invariant.
- Use the CNN classifier to predict the label and associated probability of the digits.
- Utilize non-maximum suppression to ignore overlapping and duplicate bounding boxes.
