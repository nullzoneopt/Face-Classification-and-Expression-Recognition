# Face-Recognition-and-Expression-Classification

## Libraries Needed
- sklearn
- pandas
- numpy
- openCV
- pil

## Face-Recognition
### Generate Dataset
1. Run generate_database.py file from face_recognition folder using below command.\
    python generate_database.py <person_name>
2. webcam will pop up and take 1000 pictures.
3. Make sure a folder is created inside database folder with <person_name>.
4. Repeat above process for all the persons.

### Training model
1. Open train_cv.py file from face_recognition folder.
2. Look for below line:\
   parent_directory = 
3. Give the path of the parent database folder.
4. Run the file.
5. train_cv.py file takes time to execute.
4. Alternatively you can also run train.py file.
6. Trained model would be saved under trained_models folder.

### Testing Face Recognition model
1. Open train_cv.py file from face_recognition folder.
2. Look for below line:\
   parent_directory  = 
3. Give the path of the parent database folder.
4. Look for below line:\
   model.read('....')
5. Give the path of the trained model you want to use.
6. Run the file.

## Expression Classification
### Generate Dataset
1. Run generate_database.py file from expression_recognition folder using below command.\
    python generate_database.py <expression>
    example: generate_database.py happy
2. As soon as you run this file, start smiling, webcam will pop up and take 200 pictures.
3. Repeat the above process for anger, disgust, fear, neutral, sadness, surprise
3. Make sure all folders are created inside database folder for all 7 expressions.

### Training model
1. Open train_cv.py file from face_recognition folder.
2. Look for below line:\
   parent_directory = 
3. Give the path of the parent database folder.
4. Run the file.
5. train_cv.py file takes time to execute.
4. Alternatively you can also run train.py file.
6. Trained model would be saved under trained_models folder.

### Testing Face Recognition model
1. Open test_cv.py file from face_recognition folder.
2. Look for below line:\
   parent_directory = 
3. Give the path of the parent database folder.
4. Look for below line:\
   model.read('....')
5. Give the path of the trained model you want to use.
6. Run the file.

### Integrated Model
1. Open run_on_webcam.py file, give path of both the above trained models.
face_model.read('trained_models\\face_recognition.xml')
expr_model.read('trained_models\\expression_classification.xml')
2. Give path of the database folders to create labels.\
face_database_parent_directory = 'face_recognition\\database'\
expr_database_parent_directory = 'expression_recognition\\database'
3. Run the file.
