# Project 400 - S00184337 - Geneeral Purppose Binary Classifier
# Code description
# server side code that handles the file filtration, dataframe formatting, model training, verification and model outputting
# #
# importing nessesary libraries to be used in the code below
import time
import datetime
from firebase import firebase
# from gpiozero import CPUTemperature
# import psutil
import boto3
import os
import io
import sys
import subprocess

import pandas as pd
import numpy as np

# importing the nessesary sklearn models and libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier

from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

# create the refrence to our firebase realtime database instance - this database acts as the intermediary between the webpage
# and the server side code bass
firebase = firebase.FirebaseApplication('https://project-400-64151-default-rtdb.europe-west1.firebasedatabase.app/',
                                        None)

# defining the AWS credentials
REGION = 'eu-west-1'
AWS_S3_BUCKET = 'project400'
AWS_ACCESS_KEY_ID = 'AKIA2IIIR3J4XKG452KR'
AWS_SECRET_ACCESS_KEY = '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'

# defing some variablees needed thorugh out the codebase
KEY = 'NULL'

alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


# getDataInfo function is the first function to be called at boot or if a user requests a model reboot from the webpage button
# inputs - NONE
# inputs - btnExecute, key_, names,class_0,class_1,target
# function -
# The function of the getDataInfo function, is to establish some key variables needed to search and extract the correct files
# from the AWS s3 bucket.
# The key varibales that will be extracted will have been placed on the firebase real time database from the user via the frontend
# webpage.

def getDataInfo():
    # try the following commands
    try:
        # extarct the data from these firebase db path locations & assign them to local variables
        btnExecute = firebase.put('Buttons/Execute', 'value', '0')
        key_ = firebase.get('Key/value', '')
        cols = firebase.get('NumberCols/value', '')
        # create an empty array for the column names
        names = []

        # run through the column name path locations and append the data to the list.
        names.append(firebase.get('Column Names/ColA1/value', ''))
        names.append(firebase.get('Column Names/ColB2/value', ''))
        names.append(firebase.get('Column Names/ColC3/value', ''))
        names.append(firebase.get('Column Names/ColD4/value', ''))
        names.append(firebase.get('Column Names/ColE5/value', ''))
        names.append(firebase.get('Column Names/ColF6/value', ''))
        names.append(firebase.get('Column Names/ColG7/value', ''))
        names.append(firebase.get('Column Names/ColH8/value', ''))
        names.append(firebase.get('Column Names/ColI9/value', ''))
        names.append(firebase.get('Column Names/ColJ10/value', ''))
        names.append(firebase.get('Column Names/ColK11/value', ''))
        names.append(firebase.get('Column Names/ColL12/value', ''))
        names.append(firebase.get('Column Names/ColM13/value', ''))
        names.append(firebase.get('Column Names/ColN14/value', ''))
        names.append(firebase.get('Column Names/ColO15/value', ''))
        names.append(firebase.get('Column Names/ColP16/value', ''))
        names.append(firebase.get('Column Names/ColQ17/value', ''))
        names.append(firebase.get('Column Names/ColR18/value', ''))
        names.append(firebase.get('Column Names/ColS19/value', ''))
        names.append(firebase.get('Column Names/ColT20/value', ''))

        # assign the classification labels to local values
        class_0 = (firebase.get('Binary_Classes/Class_0/value', ''))
        class_1 = (firebase.get('Binary_Classes/Class_1/value', ''))

        # assign the target value to a local variable
        target = (firebase.get('Target/Variable/value', ''))

    # if any of the above fail execute the except case.
    except:
        print('| ERROR | Issue retriving the start up data from the database |')

    # return the following variables to the codebase
    return btnExecute, key_, cols, names, class_0, class_1, target


# findFile function is a function that uses some of the key variables that we extracted using the getDataInfo to help us find
# access and extract the correct files from the s3 bucket.
# inputs - key_
# outputs - KEY
# function -
# this function is more of visual check to have a look in the s3 database that what the user has inputted as their chosen file even exsists in the s3 bucket.

def findFile():
    # create a boto s3 session
    session = boto3.Session(
        # define our AWS credentials that we assigned earlier
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    # defien what AWS resource we will be accessing
    s3 = session.resource('s3')
    # define the s3 bucket naem
    bucket = s3.Bucket(AWS_S3_BUCKET)

    # iterate through all the objects held in the s3  bucket
    for obj in bucket.objects.all():
        # print the objects for debugging purposes
        print(obj.key)
        # if the key_ value matches that of the current object being considered in the loop
        if key_ == obj.key:
            # print to the terminal
            print("FOUND IT")
            # assign that key_ value to the KEY value
            KEY = key_
        else:
            # if no match, print to the termninal
            print("NO MATCH")
    # show that KEY has been given a value
    print(KEY)
    # let the terminal know we are leaving the function
    print("Progressing ... ")
    # return the KEY var to the rest of the codebase.
    return KEY


# getData function uses all the info that has been gathered so far to be used in the following code block
# inputs - AWS credentials, KEY value
# outputs - file data frame
def getData(REGION, AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, KEY=KEY):
    # we create a boto3 client
    s3_client = boto3.client(
        # name some of the resouurces that we would like to access again and give the client our credentials
        "s3",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    # define our request responce to a local var
    response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=KEY)
    # define our request status to a local var
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    print("Status Response Code : ", status)

    # if responce 200 is true then request is a success and we progress
    if status == 200:
        print("Success responce. Status - {status | LOADING}")
        # assign the object var to the object we extract from the s3 bucket using the key value from earlier
        obj = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=KEY)
        # this obj value is the encoded and transfomred into a pandas datafrane.
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')
        # print(df)
    # if request is bad then print the error message to the termiaal
    else:
        print("Unsuccessful responce. Status - {status | FAILED}")

    # this data frame is then returned to the codebase.
    return df


# normalizeData function takes in the training data frame that has been passed though and transforms/nprmalizes the data into a
# scale from 0-1, this helps organise the data and can help with accuracy as it helps deal wit outliers in the data.
# input - training dataset
# output - transfomred training dataset.

def normalizeData(df_train):
    # define the scaler
    scaler = MinMaxScaler()
    # scale/transform the test data and hold it in a local variable
    scale_train_set = scaler.fit_transform(df_train.values)
    # create a new pandas dataframe and place the new transfromed data into it.
    train = pd.DataFrame(data=scale_train_set, columns=df_train.columns, index=df_train.index)

    # return the new transformed dataset to the rest of the codebase
    return train


# normalizeData function takes in the test data frame that has been passed though and transforms/nprmalizes the data into a
# scale from 0-1, this helps organise the data and can help with accuracy as it helps deal wit outliers in the data.
# input - test dataset
# output - transfomred test dataset.

def normalizeData(df_test):
    # define the scaler
    scaler = MinMaxScaler()
    # scale/transform the test data and hold it in a local variable
    scale_test_set = scaler.fit_transform(df_test.values)
    # create a new pandas dataframe and place the new transfromed data into it.
    test = pd.DataFrame(data=scale_test_set, columns=df_test.columns, index=df_test.index)

    # return the new transformed dataset to the rest of the codebase
    return test


# serverStats fucntion is a server side only function for debgging, it extracts hardware conditions from the server hardware
# inpouts - NONE
# outputs - cpu_temp,cpu,mem_info,disk_info
def serverStats():
    temp = CPUTemperature()
    cpu_temp = temp.temperature

    cpu = str(psutil.cpu_percent()) + '%'

    memory = psutil.virtual_memory()

    available = round(memory.available / 1024.0 / 1024.0, 1)
    total = round(memory.total / 1024.0 / 1024.0, 1)
    mem_info = str(available) + ' MB Free / ' + str(total) + 'MB total ( ' + str(memory.percent) + '% )'
    disk = psutil.disk_usage('/')
    free = round(disk.free / 1024.0 / 1024.0 / 1024.0, 1)
    total_ = round(disk.total / 1024.0 / 1024.0 / 1024.0, 1)
    disk_info = str(free) + ' GB Free / ' + str(total_) + ' GB total ( ' + str(disk.percent) + '% )'

    return cpu_temp, cpu, mem_info, disk_info


# dataFormat is a function that take the takes in the data that was extracted from the s3 bucket and formats it in to be model ready.
# inputs - data
# outputs - X,y
def dataFormat(data):
    # iterate through names of the dataframe columns
    for i in data.columns:
        # check if the target is present in the dataset
        if target in i:
            # if it is then assign the temp_y holder to that column
            temp_y = data[i]
        # if not just keep moving through
        else:
            pass

    # show if we found the target or not
    print(temp_y.head)
    # run the data through the dropna function which eliminates rowa that have null or empty values
    data.dropna(inplace=True)

    # then run the data through a loop that checks each column in the dataset
    # if any null data points have made it through will be filled with -1
    for sensor in list(data.columns)[1:-1]:
        data[sensor].fillna(-1, inplace=True)

    # we then loop through the column names
    for i in data.columns:
        # if the column name matchs that of our names array, pass on
        if i in names:
            pass
        # if the name doesnt exisit in the names array
        else:
            # then it is entirely dropped from the new dataframe
            data.drop(i, axis=1, inplace=True)
    # check out the new data.head
    print(data.head)

    # loop through the columns of the dataset
    for i in range(0, len(data.columns)):
        # for each column we rename the column var # of the step in the iteration
        data = data.rename(columns={data.columns[i]: 'var_{}'.format(i)})

    # check out the new head
    print(data.head())

    # create a label encoder - this is needed for categorical data that needs to be converted from labesl to numerical form
    le = LabelEncoder()
    #
    # loop thirugh the dataset
    for i in data:
        # check if the column in consideration is of object type - i.e has labeled data
        if data[i].dtype == object:
            # if it has, then encode the labeled data weithin that column
            data[i] = le.fit_transform(data[i].values)
        else:
            # if it is anythng other than an object data type then pass on
            pass

    # check if the data that will be y(predictor) is of object type
    if temp_y.dtype == object:
        # if it is then action is required
        print('action - labels to be encoded')
        # The labels inside the column will be encoded to 0 and 1
        temp_y = le.fit_transform(temp_y.values)
    else:
        pass

        # we then append the new y data back into the data data frame
    data['label'] = temp_y
    # check out the updated dataframe
    print(data.info)

    from sklearn.ensemble import ExtraTreesClassifier
    # assign the new y dataframe as the encoded label column
    y = data['label']
    print(y)
    # eveeytiong but the label column as the new X datafrane
    X = data.drop(columns='label')
    print(X)

    # return the new X and y dataframe back to the codebaae
    return X, y


# modelOps function is to handle the model work, it splits the data, fits it to the model and trains the alogorithim
# inputs - X and y
# outputs trained ML algorithim
def modelOps(X, y):
    # we use the sklearn train test split our dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print('test state')

    # pass the x training set through the normalization function
    X_train = normalizeData(X_train)
    print(X_train)
    # pass the x test set through the normalization function
    X_test = normalizeData(X_test)
    print(X_test)

    # create the random forest ml model
    forest_model = RandomForestClassifier(n_estimators=25, criterion='gini', max_depth=5, random_state=21, n_jobs=-1)

    # fir the data to the model
    forest_model.fit(X_train, y_train)

    # get trainign and test scores and defien a report to isolate scores later
    # get the training and test score as 0-100% value to 2 decimal places
    training_score = round((forest_model.score(X_train, y_train) * 100.0), 2)
    test_score = round((forest_model.score(X_test, y_test) * 100.0), 2)

    # create a report variable extracted from the model
    report = (classification_report(y_test, forest_model.predict(X_test), output_dict=True))

    # extract isolated report macro scores to be passed to the webpage
    precision = round(((report['macro avg']['precision']) * 100.0), 2)
    recall = round(report['macro avg']['recall'], 2)
    f1 = round(report['macro avg']['f1-score'], 2)
    support = round(report['macro avg']['support'], 2)

    # push the data to teh firbase datababe
    try:
        db = firebase.put('Model Scoring', 'Training Score', training_score)
        db = firebase.put('Model Scoring', 'Test Score', test_score)
        db = firebase.put('Model Scoring', 'Precision', precision)
        db = firebase.put('Model Scoring', 'Recall', recall)
        db = firebase.put('Model Scoring', 'f1-score', f1)
        db = firebase.put('Model Scoring', 'Support', support)
    except:
        pass
    # pass the trained model back to the codebase
    return forest_model


# pred is the function that provides the final predtion from the trained model
# inputs - the trained model and the x data
# outputs - the probability of the two classes
def pred(model, x):
    try:
        # make a prediction using the passed model and x values
        y_pred = model.predict(x)
        print(y_pred)

    except:
        print('| ERROR | |')

    try:
        # predict the propability of each class from the model and x
        prob = model.predict_proba(x)
        print(prob)
    except:
        print('case2')

    # the  output of the predict_proba fucntion produxes an array [class0,class1]
    # isolate the class 0 proability value
    prob_0 = prob.item(0)
    # isolate the class 1 probability value
    prob_1 = prob.item(1)

    # pass back the proability of both classes back to the codebase
    return prob_0, prob_1


# clearAll fucntion reacts to the button push via the webpage, it clears all the model metrics back to 0
def clearAll():
    try:
        db = firebase.put('Model Scoring', 'Training Score', '0')
        db = firebase.put('Model Scoring', 'Test Score', '0')
        db = firebase.put('Model Scoring', 'Precision', '0')
        db = firebase.put('Model Scoring', 'Recall', '0')
        db = firebase.put('Model Scoring', 'f1-score', '0')
        db = firebase.put('Model Scoring', 'Support', '0')
        db = firebase.put('Algorithim Output', 'Class_0 Probability', '0')
        db = firebase.put('Algorithim Output', 'Class_1 Probability', '0')


    except:
        pass


# finalOps is the final loop that continually produces teh model output based on the slider values
def finalOp(cols):
    # first we get all the slider values from the db and sclae the value down to fit the data i.e between 0-1
    try:
        val1 = firebase.get('Sliders/Slider_1/value', '')
        val1 = ((float(val1)) / 100.0)
        # print(val1)
        val2 = firebase.get('Sliders/Slider_2/value', '')
        val2 = ((float(val2)) / 100.0)
        # print(val2)
        val3 = firebase.get('Sliders/Slider_3/value', '')
        val3 = ((float(val3)) / 100.0)
        # print(val3)
        val4 = firebase.get('Sliders/Slider_4/value', '')
        val4 = ((float(val4)) / 100.0)
        # print(val4)
        val5 = firebase.get('Sliders/Slider_5/value', '')
        val5 = ((float(val5)) / 100.0)
        # print(val5)
        val6 = firebase.get('Sliders/Slider_6/value', '')
        val6 = ((float(val6)) / 100.0)
        # print(val6)
        val7 = firebase.get('Sliders/Slider_7/value', '')
        val7 = ((float(val7)) / 100.0)
        # print(val7)
        val8 = firebase.get('Sliders/Slider_8/value', '')
        val8 = ((float(val8)) / 100.0)
        # print(val8)
        val9 = firebase.get('Sliders/Slider_9/value', '')
        val9 = ((float(val9)) / 100.0)
        # print(val9)
        val10 = firebase.get('Sliders/Slider_10/value', '')
        val10 = ((float(val10)) / 100.0)
        # print(val10)
        val11 = firebase.get('Sliders/Slider_11/value', '')
        val11 = ((float(val11)) / 100.0)
        # print(val11)
        val12 = firebase.get('Sliders/Slider_12/value', '')
        val12 = ((float(val12)) / 100.0)
        # print(val12)
        val13 = firebase.get('Sliders/Slider_13/value', '')
        val13 = ((float(val13)) / 100.0)
        # print(val13)
        val14 = firebase.get('Sliders/Slider_14/value', '')
        val14 = ((float(val14)) / 100.0)
        # print(val14)
        val15 = firebase.get('Sliders/Slider_15/value', '')
        val15 = ((float(val15)) / 100.0)
        # print(val15)
        val16 = firebase.get('Sliders/Slider_16/value', '')
        val16 = ((float(val16)) / 100.0)
        # print(val16)
        val17 = firebase.get('Sliders/Slider_17/value', '')
        val17 = ((float(val17)) / 100.0)
        # print(val17)
        val18 = firebase.get('Sliders/Slider_18/value', '')
        val18 = ((float(val18)) / 100.0)
        # print(val18)
        val19 = firebase.get('Sliders/Slider_19/value', '')
        val19 = ((float(val19)) / 100.0)
        # print(val19)
        val20 = firebase.get('Sliders/Slider_20/value', '')
        val20 = ((float(val20)) / 100.0)
        # print(val20)
    except:
        print("| ERROR | Issue in retriving the slider values |")

    # create a temp dataframe then pass the slider values to the columns in the 1st row
    holderdf = pd.DataFrame([[val1, val2, val3,
                              val4, val5, val6,
                              val7, val8, val9,
                              val10, val11, val12,
                              val13, val14, val15,
                              val16, val17, val18,
                              val19, val20]], columns=('var_0', 'var_1', 'var_2',
                                                       'var_3', 'var_4', 'var_5',
                                                       'var_6', 'var_7', 'var_8',
                                                       'var_9', 'var_10', 'var_11',
                                                       'var_12', 'var_13', 'var_14',
                                                       'var_15', 'var_16', 'var_17',
                                                       'var_18', 'var_19'))

    try:
        # check out the input data frame
        print(holderdf)
        # if the amount of cols is less than 20 proceed
        if (int(cols) < 20):
            # create the indec place holder
            N = 20 - int(cols)
            print(N)
            # create a new dataframe based on the number of columns the user has inputted in the webpage
            testdf = holderdf.iloc[:, :-N]
        else:
            # else just fill the new datafrane with the temporary datafraem
            testdf = holderdf
            print(testdf)
    except:

        print("| ERROR | Issue the test data frame |")

    try:
        # call the pred fucntion to get the two classes probability
        ml_op0, ml_op1 = pred(forest_model, testdf)

        # make these prpbabilities to 0-100% values rounded to 2 decimal percenatges
        ml_op0 = round((ml_op0 * 100), 2)
        ml_op1 = round((ml_op1 * 100), 2)

        print(ml_op0)
        print(ml_op1)
    except:
        print('| ERROR | Conflictions in in the prediction function |')

    # call some debugging functions
    # get the cpu stats
    # cpu_temp,cpu,mem_info,disk_info = serverStats()
    # get a timestamp on each iteration
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    try:
        # place all this date on the db to be picked up by the webpage
        # db = firebase.put('Server Condition/CPU Temperature','Degree Celcius',cpu_temp)
        # db = firebase.put('Server Condition/CPU Usage','Percentage',cpu)
        # db = firebase.put('Server Condition/CPU Memory Info','MegaBytes',mem_info)
        # db = firebase.put('Server Condition/CPU Disk Info','GigaBytes',disk_info)
        db = firebase.put('Algorithim Output', 'Class_0 Probability', ml_op0)
        db = firebase.put('Algorithim Output', 'Class_1 Probability', ml_op1)
        db = firebase.put('Algorithim Output', 'Chance Of Failure', ml_op1)
        db = firebase.put('Timestamp', 'value', timestamp)

    except:
        print('| ERROR | Issues loading the server conditions and algorithim outputs to the database |')

    print("| STATUS REPORT | Successful iteration of the finalOp function |")


# sliderLabel is function to find annd route teh column names to the indexed slider
def sliderLabel(cols, names):
    # create a alphabet array
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

    # iterate as long as the length of list of names
    for i in range(0, (len(names))):
        # print('{}{}'.format(alpha[i],(i+1)))

        if i > (int(cols) - 1):
            names[i] = ''
            db = firebase.put('Column Names', 'Col{}{}'.format(alpha[i], (i + 1)), '')
        elif (int(cols) == 1):
            names[0] = ''
        else:
            pass


##### AT BOOT #####
# when system is started, run through all the key start up functions
# when this is processed ok we moved into the while loop

print(
    '| STATUS REPORT | Running essential startup, file extarction, data filtration, formatting and model fitting functions |')
#btnExecute, key_, cols, names, class_0, class_1, target = getDataInfo()
#KEY = findFile()
#data = getData(REGION, AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, KEY)
#sliderLabel(cols, names)
#X, y = dataFormat(data)
#forest_model = modelOps(X, y)

error = '| ERROR | SERVER - None |'

while True:

    try:
        btnExecute = firebase.get('Buttons/Execute/value', '')
        btnClear = firebase.get('Buttons/Clear/value', '')

    except:
        pass
    # if the reboot button is pushed then we reestart the code and call all the nessesary start up functions
    if int(btnExecute) == 1:
        print(
            '| STATUS REPORT | EVENT - webpage button push | Restarting essential startup, file extarction, data filtration, formatting and model fitting functions |')
        try:
            btnExecute, key_, cols, names, class_0, class_1, target = getDataInfo()
        except:
            error = '| ERROR | SERVER - Failed to aquire key data |'
        try:
            KEY = findFile()
        except:
            error = '| ERROR | SERVER - Failed to find file in storage |'

        try:
            data = getData(REGION, AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, KEY)
        except:
            error = '| ERROR | SERVER - Failed to pull file from storage |'

        try:
            sliderLabel(cols, names)
        except:
            error = '| ERROR | SERVER - Failed to aquire slider data |'
        try:
            X, y = dataFormat(data)
        except:
            error = '| ERROR | SERVER - Failed to format dataframe |'
        try:
            forest_model = modelOps(X, y)
        except:
            error = '| ERROR | SERVER - Failed in model fitting |'
        print('| STATUS REPORT | EVENT - webpage button push | Restarting session successful |')

    # if the button is not pushed, keep l;oopign through the final ops whioch handles the live predictions and slider control.
    elif int(btnExecute) == 0:
        try:
            finalOp(cols)
        except:
            error = '| ERROR | SERVER - Failed to extract model output |'

    if int(btnClear) == 1:
        print('| STATUS REPORT | EVENT - webpage button push | Clearing all parameters |')
        clearAll()
    else:
        pass



