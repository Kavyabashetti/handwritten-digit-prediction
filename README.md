# handwritten-digit-prediction
"output_type": "execute_result"
    }
   ],
   "source": [
    "# Initialize the logistic regression model\n",
    "model = LogisticRegression()\n",
    "\n",
    "# Train the model on the training data\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e8bf6e3",
   "metadata": {},
   "source": [
    "## Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b416f5a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9722222222222222\n",
      "Confusion Matrix:\n",
      "[[33  0  0  0  0  0  0  0  0  0]\n",
      " [ 0 28  0  0  0  0  0  0  0  0]\n",
      " [ 0  0 33  0  0  0  0  0  0  0]\n",
      " [ 0  0  0 33  0  1  0  0  0  0]\n",
      " [ 0  1  0  0 45  0  0  0  0  0]\n",
      " [ 0  0  0  0  0 44  1  0  0  2]\n",
      " [ 0  0  0  0  0  1 34  0  0  0]\n",
      " [ 0  0  0  0  0  0  0 33  0  1]\n",
      " [ 0  0  0  0  0  1  0  0 29  0]\n",
      " [ 0  0  0  1  0  0  0  0  1 38]]\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        33\n",
      "           1       0.97      1.00      0.98        28\n",
      "           2       1.00      1.00      1.00        33\n",
      "           3       0.97      0.97      0.97        34\n",
      "           4       1.00      0.98      0.99        46\n",
      "           5       0.94      0.94      0.94        47\n",
      "           6       0.97      0.97      0.97        35\n",
      "           7       1.00      0.97      0.99        34\n",
      "           8       0.97      0.97      0.97        30\n",
      "           9       0.93      0.95      0.94        40\n",
      "\n",
      "    accuracy                           0.97       360\n",
      "   macro avg       0.97      0.97      0.97       360\n",
      "weighted avg       0.97      0.97      0.97       360\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Make predictions on the test data\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate the model's accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "\n",
    "# Generate confusion matrix and classification report\n",
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "class_report = classification_report(y_test, y_pred)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(conf_matrix)\n",
    "print(\"Classification Report:\")\n",
    "print(class_report)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f689e9f",
   "metadata": {},
   "source": [
    "## Prediction "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "81d839d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Digit: 4\n"
     ]
    }
   ],
   "source": [
    "from PIL import Image\n",
    "\n",
    "# Load the image and convert it to grayscale\n",
    "image = Image.open('download.png').convert('L')\n",
    "\n",
    "# Resize the image to 8x8 pixels (the same size as the dataset images)\n",
    "image = image.resize((8, 8))\n",
    "\n",
    "# Convert the image to a numpy array\n",
    "new_image = np.array(image)\n",
    "\n",
    "# Flatten the 8x8 image into a 1D array for prediction\n",
    "new_image_flattened = new_image.flatten().reshape(1, -1)\n",
    "\n",
    "# Scale the feature variables using the previously defined scaler\n",
    "new_image_scaled = scaler.transform(new_image_flattened)\n",
    "\n",
    "# Make a prediction using the model\n",
    "predicted_digit = model.predict(new_image_scaled)\n",
    "\n",
    "print(\"Predicted Digit:\", predicted_digit[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "414cfef6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
