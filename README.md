## Plant_seed_classification

 # Step: 
       1. main.py (epoch,train_Path,test_path is fixed):
            run(train_Path, epoch), return class_idx(class library: include name and index)
            test(test_path)
       2. resnet50_model.py:
            run(): let main.py use
            load(): load dataset
            split(): split data to train and valid
            module(): set model, device and optimizer
            train(): train model and save loss and accurancy image
            valid(): caculate loss value and Number of correct classifications
            test(): predict the classification of photos
 # Result:
       1. accurancy.png: each epoch time's valid accurancy and train accurancy
       2. loss.png: each epoch time's valid loss and train loss
       3. submission.csv: record the predict of test image set's classifications
       4. kaggle_score.png: submission.csv result score in kaggle
